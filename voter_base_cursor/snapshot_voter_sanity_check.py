"""snapshot_voter_sanity_check.py

Task 4 — Eligible voter sanity check for a single Snapshot proposal.

Checks whether all Snapshot voters for a proposal also held a positive
balance of a given ERC20 governance token at the proposal's snapshot block.

Environment variables (must be set outside this script):
  - RPC_URL: Ethereum RPC URL (optional here but logged, e.g. Alchemy mainnet)
  - ETHERSCAN_API_KEY: Etherscan API key (required)

Example usage (PowerShell):

  $env:RPC_URL="https://eth-mainnet.g.alchemy.com/v2/xxxxx"
  $env:ETHERSCAN_API_KEY="your_etherscan_key"
  python snapshot_voter_sanity_check.py `
    --proposal_id 0xfcd429c8fcb5fc44a0bea9bf078726ef48b1c76ca1039a8c6c9dff23f4547e30 `
    --token 0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9

Example usage (bash):

  export RPC_URL="https://eth-mainnet.g.alchemy.com/v2/xxxxx"
  export ETHERSCAN_API_KEY="your_etherscan_key"
  python snapshot_voter_sanity_check.py \\
    --proposal_id 0xfcd429c8fcb5fc44a0bea9bf078726ef48b1c76ca1039a8c6c9dff23f4547e30 \\
    --token 0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Iterable, Tuple, Set

import requests


SNAPSHOT_GRAPHQL = "https://hub.snapshot.org/graphql"
ETHERSCAN_API = "https://api.etherscan.io/api"

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def snapshot_graphql(query: str, variables: dict) -> dict:
    """Call Snapshot GraphQL with basic retry on 5xx/429/timeouts.

    Returns the `data` field or raises RuntimeError with a descriptive message.
    """
    max_retries = 5
    backoff_base = 2.0
    timeout_s = 60

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                SNAPSHOT_GRAPHQL,
                json={"query": query, "variables": variables},
                timeout=timeout_s,
            )
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise RuntimeError(
                    f"Retryable GraphQL HTTP {r.status_code}: {(r.text or '')[:200]}"
                )
            if r.status_code >= 400:
                raise RuntimeError(
                    f"GraphQL HTTP {r.status_code}: {(r.text or '')[:200]}"
                )
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"Snapshot GraphQL errors: {data['errors']}")
            return data["data"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= max_retries - 1:
                break
            sleep_s = backoff_base * (2 ** attempt)
            time.sleep(sleep_s)

    raise RuntimeError(f"Snapshot GraphQL failed after retries: {last_err}") from last_err


def fetch_proposal_snapshot_block(proposal_id: str) -> Tuple[int, str]:
    """Return (snapshot_block, space_id) for a proposal."""
    q = """
    query Proposal($id: String!) {
      proposal(id: $id) {
        id
        snapshot
        space { id }
        type
        choices
      }
    }
    """
    data = snapshot_graphql(q, {"id": proposal_id})
    p = data.get("proposal")
    if not p:
        raise RuntimeError(f"Proposal not found on Snapshot: {proposal_id}")
    try:
        snapshot_block = int(p["snapshot"])
    except (KeyError, ValueError, TypeError) as e:  # noqa: BLE001
        raise RuntimeError(f"Invalid snapshot value in proposal: {p!r}") from e
    space_id = str(p.get("space", {}).get("id", ""))
    return snapshot_block, space_id


def fetch_all_snapshot_voters(proposal_id: str, page_size: int = 1000) -> Set[str]:
    """Fetch all voters for a proposal (lowercased)."""
    voters: Set[str] = set()
    skip = 0

    q = """
    query Votes($proposal: String!, $first: Int!, $skip: Int!) {
      votes(
        first: $first
        skip: $skip
        where: { proposal: $proposal }
        orderBy: "created"
        orderDirection: asc
      ) {
        id
        voter
        created
        choice
        vp
      }
    }
    """

    while True:
        data = snapshot_graphql(
            q,
            {"proposal": proposal_id, "first": page_size, "skip": skip},
        )
        batch = data.get("votes") or []
        if not batch:
            break
        for v in batch:
            addr = (v.get("voter") or "").strip().lower()
            if addr:
                voters.add(addr)
        if len(batch) < page_size:
            break
        skip += page_size
    return voters


def etherscan_tokentx_pages(
    token: str,
    snapshot_block: int,
    page_size: int,
    sleep_s: float,
    max_pages: int,
    api_key: str,
) -> Iterable[List[dict]]:
    """Yield pages of ERC20 transfer events from Etherscan up to snapshot_block."""
    token = token.lower()
    for page in range(1, max_pages + 1):
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": token,
            "startblock": 0,
            "endblock": snapshot_block,
            "page": page,
            "offset": page_size,
            "sort": "asc",
            "apikey": api_key,
        }
        try:
            r = requests.get(ETHERSCAN_API, params=params, timeout=60)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Etherscan request failed on page {page}: {e}") from e

        if r.status_code >= 400:
            raise RuntimeError(
                f"Etherscan HTTP {r.status_code} on page {page}: {(r.text or '')[:200]}"
            )
        payload = r.json()
        status = payload.get("status")
        message = payload.get("message", "")
        result = payload.get("result")

        # "No transactions found" -> stop cleanly
        if status == "0" and isinstance(result, str) and "No transactions" in result:
            break

        if status not in ("0", "1"):
            raise RuntimeError(
                f"Etherscan unexpected status on page {page}: {payload!r}"
            )

        if not isinstance(result, list) or not result:
            # empty list => done
            break

        yield result

        if len(result) < page_size:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)


def build_balances_from_transfers(
    pages: Iterable[List[dict]],
) -> Dict[str, int]:
    """Replay ERC20 Transfer logs into balances mapping (raw integer units)."""
    balances: Dict[str, int] = {}

    for batch in pages:
        for tx in batch:
            from_addr = (tx.get("from") or "").strip().lower()
            to_addr = (tx.get("to") or "").strip().lower()
            value_str = tx.get("value") or "0"
            try:
                value = int(value_str)
            except ValueError:
                continue
            if value <= 0:
                continue

            # Subtract from sender (except mint from zero)
            if from_addr and from_addr != ZERO_ADDRESS:
                balances[from_addr] = balances.get(from_addr, 0) - value

            # Add to receiver (except burn to zero)
            if to_addr and to_addr != ZERO_ADDRESS:
                balances[to_addr] = balances.get(to_addr, 0) + value

    return balances


def holders_from_balances(balances: Dict[str, int]) -> Set[str]:
    """Return addresses with strictly positive balance."""
    return {addr for addr, bal in balances.items() if bal > 0}


def compute_overlap_stats(
    voters: Set[str],
    holders: Set[str],
) -> Tuple[int, int, int, float]:
    n_voters = len(voters)
    n_holders = len(holders)
    overlap = len(voters & holders)
    overlap_ratio = (overlap / n_voters) if n_voters > 0 else 0.0
    return n_voters, n_holders, overlap, overlap_ratio


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sanity check: are all Snapshot voters token holders at snapshot?"
    )
    parser.add_argument(
        "--proposal_id",
        required=True,
        help="Snapshot proposal id (0x...)",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Governance token contract address (0x...) on Ethereum mainnet",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=10_000,
        help="Etherscan pagination size (default: 10000)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Sleep in seconds between Etherscan pages (default: 0.25)",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=2000,
        help="Maximum number of Etherscan pages to fetch (default: 2000)",
    )
    args = parser.parse_args(argv)

    rpc_url = os.environ.get("RPC_URL")
    etherscan_key = os.environ.get("ETHERSCAN_API_KEY")

    if not etherscan_key:
        raise RuntimeError(
            "Missing ETHERSCAN_API_KEY environment variable. "
            "Set it before running this script."
        )

    print("=== Snapshot voter sanity check ===")
    print(f"Proposal id   : {args.proposal_id}")
    print(f"Token address : {args.token}")
    print(f"RPC_URL set   : {'yes' if rpc_url else 'no'}")
    print()

    # 1) Fetch proposal metadata and snapshot block
    snapshot_block, space_id = fetch_proposal_snapshot_block(args.proposal_id)
    print(f"Snapshot space   : {space_id or '(unknown)'}")
    print(f"Snapshot block   : {snapshot_block}")

    # 2) Fetch all voters from Snapshot
    print("Fetching Snapshot voters ...")
    snapshot_voters = fetch_all_snapshot_voters(args.proposal_id)
    print(f"  n_snapshot_voters = {len(snapshot_voters)}")

    # 3) Reconstruct token balances at snapshot block via Etherscan ERC20 transfers
    print("Fetching Etherscan token transfer pages and replaying balances ...")
    pages = etherscan_tokentx_pages(
        token=args.token,
        snapshot_block=snapshot_block,
        page_size=args.page_size,
        sleep_s=args.sleep,
        max_pages=args.max_pages,
        api_key=etherscan_key,
    )
    balances = build_balances_from_transfers(pages)
    holders = holders_from_balances(balances)
    print(f"  n_token_holders_at_snapshot = {len(holders)}")

    # 4) Sanity check: voters subset of holders
    missing = snapshot_voters - holders
    n_voters, n_holders, overlap, overlap_ratio = compute_overlap_stats(
        snapshot_voters, holders
    )

    print()
    print("=== Summary ===")
    print(f"n_snapshot_voters           : {n_voters}")
    print(f"n_token_holders_at_snapshot : {n_holders}")
    # Avoid non-ASCII symbols so Windows consoles with cp1252 don't fail.
    print(f"overlap (voters INTERSECT holders) : {overlap}")
    print(f"overlap_ratio               : {overlap_ratio:.6f}")

    if not missing:
        print()
        print("PASS: All Snapshot voters appear in the token holder set at snapshot block.")
        return

    print()
    print(
        f"FAIL: {len(missing)} Snapshot voters do NOT appear as token holders "
        "at the snapshot block."
    )

    missing_path = "missing_snapshot_voters.txt"
    with open(missing_path, "w", encoding="utf-8") as f:
        for addr in sorted(missing):
            f.write(addr + "\n")

    print(f"Missing voter addresses written to: {missing_path}")
    print()
    print("First 10 missing voters:")
    for addr in list(sorted(missing))[:10]:
        print(f"  - {addr}")

    # Nice-to-have diagnostics
    print()
    print("Likely reasons for missing voters include:")
    print("  - Snapshot strategy not pure token balance (delegation/staking/LP wrapper).")
    print("  - Wrong governance token contract address for this space/proposal.")
    print("  - Proposal may be on an L2 network, not Ethereum mainnet.")
    print("  - Snapshot strategies pulling voting power from another source.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

