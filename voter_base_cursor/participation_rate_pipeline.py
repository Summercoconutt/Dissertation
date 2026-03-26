"""participation_rate_pipeline.py

Research pipeline to compute proposal-level and DAO-level participation rates
for Snapshot spaces, following the methodology in participation_rate_measures.txt.

Scope of this first version
---------------------------
- Fully supports ONLY simple ERC20 balance-based strategies on Ethereum mainnet.
  We approximate eligibility as:
      eligible_voter = address with ERC20 balance > 0 at the proposal snapshot block
  reconstructed via ERC20 Transfer logs from Etherscan.
- For all other strategy types (delegation, staking, LP, multi-strategy, L2, etc.)
  we do NOT fabricate eligibility. Those proposals/spaces are flagged as unsupported.

Environment
-----------
- ETHERSCAN_API_KEY: required for supported ERC20 strategies.
- RPC_URL: optional (not used in v1, but logged for transparency).

Outputs
-------
1) proposal_participation_metrics.csv
   - space, proposal_id, snapshot_block, strategy_types, support_status, support_reason,
     n_actual_voters, n_eligible_voters, participation_rate

2) dao_participation_metrics.csv
   - space, mean_participation_rate, median_participation_rate,
     weighted_mean_participation_rate (currently equal-weighted),
     n_supported_proposals, n_unsupported_proposals, support_ratio

3) strategy_summary.csv
   - space, detected_strategies, reconstruction_supported, reconstruction_reason
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests


SNAPSHOT_GRAPHQL = "https://hub.snapshot.org/graphql"
ETHERSCAN_API = "https://api.etherscan.io/api"


# =========================
# Snapshot helpers
# =========================


def snapshot_graphql(query: str, variables: Optional[dict] = None, timeout_s: int = 60) -> dict:
    """Call Snapshot GraphQL with basic retry on 5xx/429/timeouts."""
    max_retries = 5
    backoff_base = 2.0

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            payload = {"query": query}
            if variables is not None:
                payload["variables"] = variables
            r = requests.post(SNAPSHOT_GRAPHQL, json=payload, timeout=timeout_s)
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
            time.sleep(backoff_base * (2**attempt))
    raise RuntimeError(f"Snapshot GraphQL failed after retries: {last_err}") from last_err


def fetch_space_proposals(space: str, first: int) -> List[dict]:
    """Fetch up to `first` most recent proposals for a space, including strategies."""
    q = """
    query Proposals($space: String!, $first: Int!) {
      proposals(
        first: $first
        where: { space_in: [$space] }
        orderBy: "created"
        orderDirection: desc
      ) {
        id
        title
        snapshot
        created
        space { id }
        strategies { name network params }
      }
    }
    """
    data = snapshot_graphql(q, {"space": space, "first": first})
    return data.get("proposals") or []


def fetch_all_votes(proposal_id: str, page_size: int = 1000, max_pages: int = 200) -> List[dict]:
    """Fetch all votes for a proposal (paginated)."""
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
    rows: List[dict] = []
    skip = 0
    for _ in range(max_pages):
        data = snapshot_graphql(q, {"proposal": proposal_id, "first": page_size, "skip": skip})
        batch = data.get("votes") or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        skip += page_size
    return rows


def get_unique_voters(votes: List[dict]) -> Set[str]:
    """Extract unique lowercased voter addresses from Snapshot votes."""
    out: Set[str] = set()
    for v in votes:
        addr = (v.get("voter") or "").strip().lower()
        if addr:
            out.add(addr)
    return out


# =========================
# Strategy detection
# =========================


def classify_space_strategies(strategies: List[dict]) -> Tuple[str, bool, str, Optional[str]]:
    """Classify strategy set for a space.

    Returns:
        (strategy_types_str, reconstruction_supported, reconstruction_reason, token_address_if_simple)
    """
    if not strategies:
        return ("none", False, "no strategies returned from Snapshot", None)

    names = [str(s.get("name") or "").strip() for s in strategies]
    strategy_types_str = ",".join(sorted(set(names)))

    # Simple ERC20 balance on Ethereum mainnet
    SIMPLE_NAMES = {"erc20-balance-of", "erc20-with-balance"}

    if len(strategies) == 1 and names[0] in SIMPLE_NAMES:
        s0 = strategies[0]
        params = s0.get("params") or {}
        token_addr = params.get("address") or params.get("token")
        network = str(s0.get("network") or "1")
        if not token_addr:
            return (
                strategy_types_str,
                False,
                "simple ERC20 strategy but missing token address in params",
                None,
            )
        if network != "1":
            return (
                strategy_types_str,
                False,
                f"simple ERC20 strategy but network={network} != 1 (ETH mainnet)",
                None,
            )
        return (
            strategy_types_str,
            True,
            "simple ERC20 balance-of on Ethereum mainnet",
            str(token_addr),
        )

    # Everything else is unsupported in v1
    return (
        strategy_types_str,
        False,
        "complex or multi-strategy configuration; eligibility reconstruction not implemented in v1",
        None,
    )


# =========================
# Etherscan ERC20 helper
# =========================


def etherscan_tokentx_pages(
    token: str,
    snapshot_block: int,
    page_size: int,
    sleep_s: float,
    max_pages: int,
    api_key: str,
) -> Iterable[List[dict]]:
    """Yield pages of ERC20 transfer events from Etherscan up to snapshot_block."""
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
        result = payload.get("result")

        if status == "0" and isinstance(result, str) and "No transactions" in result:
            break
        if status not in ("0", "1"):
            raise RuntimeError(f"Etherscan unexpected status on page {page}: {payload!r}")
        if not isinstance(result, list) or not result:
            break

        yield result

        if len(result) < page_size:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)


def balances_from_transfers(pages: Iterable[List[dict]]) -> Dict[str, int]:
    """Replay ERC20 Transfer logs into raw integer balances."""
    ZERO = "0x0000000000000000000000000000000000000000"
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
            if from_addr and from_addr != ZERO:
                balances[from_addr] = balances.get(from_addr, 0) - value
            if to_addr and to_addr != ZERO:
                balances[to_addr] = balances.get(to_addr, 0) + value
    return balances


def holders_from_balances(balances: Dict[str, int]) -> Set[str]:
    """Addresses with strictly positive balance."""
    return {addr for addr, bal in balances.items() if bal > 0}


# =========================
# Participation computation
# =========================


def compute_proposal_metrics_for_space(
    space: str,
    proposals: List[dict],
    etherscan_key: Optional[str],
    page_size: int,
    sleep_s: float,
    max_pages: int,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Compute proposal-level metrics and per-space strategy summary."""
    proposal_rows: List[dict] = []
    strategy_summary: Dict[str, dict] = {}

    # Strategy detection uses the strategies on each proposal; assume stable across proposals.
    all_strats = proposals[0].get("strategies") if proposals else []
    strat_types, supported, reason, token_addr = classify_space_strategies(all_strats)
    strategy_summary[space] = {
        "space": space,
        "detected_strategies": strat_types,
        "reconstruction_supported": supported,
        "reconstruction_reason": reason,
        "token_address_if_simple": token_addr or "",
    }

    for p in proposals:
        pid = p["id"]
        snapshot_block = int(p["snapshot"])
        votes = fetch_all_votes(pid)
        voters = get_unique_voters(votes)

        row = {
            "space": space,
            "proposal_id": pid,
            "snapshot_block": snapshot_block,
            "strategy_types": strat_types,
            "support_status": "supported" if supported else "unsupported",
            "support_reason": reason,
            "n_actual_voters": len(voters),
            "n_eligible_voters": None,
            "participation_rate": None,
        }

        if not supported:
            proposal_rows.append(row)
            continue

        if not etherscan_key:
            row["support_status"] = "unsupported"
            row["support_reason"] = "ETHERSCAN_API_KEY missing"
            proposal_rows.append(row)
            continue

        # Reconstruct eligible voters via ERC20 balances at snapshot block.
        try:
            pages = etherscan_tokentx_pages(
                token=token_addr,
                snapshot_block=snapshot_block,
                page_size=page_size,
                sleep_s=sleep_s,
                max_pages=max_pages,
                api_key=etherscan_key,
            )
            balances = balances_from_transfers(pages)
            holders = holders_from_balances(balances)
            n_eligible = len(holders)
            row["n_eligible_voters"] = n_eligible
            if n_eligible > 0:
                row["participation_rate"] = len(voters) / n_eligible
            else:
                row["participation_rate"] = None
                row["support_status"] = "partially_supported"
                row["support_reason"] = "no eligible holders reconstructed (0 balances)"
        except Exception as e:  # noqa: BLE001
            row["support_status"] = "unsupported"
            row["support_reason"] = f"error during eligibility reconstruction: {e}"

        proposal_rows.append(row)

    return proposal_rows, strategy_summary


def aggregate_dao_metrics(proposal_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate proposal-level metrics to DAO/space-level metrics."""
    rows: List[dict] = []
    for space, g in proposal_df.groupby("space"):
        supported_mask = g["support_status"].isin(["supported", "partially_supported"])
        supported = g.loc[supported_mask & g["participation_rate"].notna(), "participation_rate"]
        n_supported = int(supported_mask.sum())
        n_unsupported = int((~supported_mask).sum())
        support_ratio = n_supported / (n_supported + n_unsupported) if (n_supported + n_unsupported) > 0 else 0.0

        if not supported.empty:
            mean_pr = float(supported.mean())
            median_pr = float(supported.median())
            # For now, weighted mean == unweighted mean (no weights yet).
            w_mean_pr = mean_pr
        else:
            mean_pr = float("nan")
            median_pr = float("nan")
            w_mean_pr = float("nan")

        rows.append(
            {
                "space": space,
                "mean_participation_rate": mean_pr,
                "median_participation_rate": median_pr,
                "weighted_mean_participation_rate": w_mean_pr,
                "n_supported_proposals": n_supported,
                "n_unsupported_proposals": n_unsupported,
                "support_ratio": support_ratio,
            }
        )
    return pd.DataFrame(rows)


# =========================
# CLI
# =========================


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute proposal- and DAO-level participation rates for Snapshot spaces."
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        required=True,
        help="Snapshot space ids (e.g. aavedao.eth uniswapgovernance.eth)",
    )
    parser.add_argument(
        "--max_proposals",
        type=int,
        default=10,
        help="Max number of most recent proposals per space to process.",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=10_000,
        help="Etherscan pagination size for ERC20 transfers (default: 10000).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Sleep in seconds between Etherscan pages (default: 0.25).",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=2_000,
        help="Maximum number of Etherscan pages per proposal (default: 2000).",
    )
    parser.add_argument(
        "--out_dir",
        default="out_participation",
        help="Output directory for CSVs (default: out_participation).",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rpc_url = os.environ.get("RPC_URL")
    etherscan_key = os.environ.get("ETHERSCAN_API_KEY")

    print("=== Participation rate pipeline ===")
    print(f"Spaces            : {', '.join(args.spaces)}")
    print(f"Max proposals     : {args.max_proposals}")
    print(f"RPC_URL set       : {'yes' if rpc_url else 'no'}")
    print(f"ETHERSCAN_API_KEY : {'yes' if etherscan_key else 'no'}")
    print()

    all_proposal_rows: List[dict] = []
    strategy_rows: List[dict] = []

    for space in args.spaces:
        print(f"--- Space: {space} ---")
        try:
            proposals = fetch_space_proposals(space, args.max_proposals)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR fetching proposals: {e}")
            # Record as fully unsupported
            strategy_rows.append(
                {
                    "space": space,
                    "detected_strategies": "",
                    "reconstruction_supported": False,
                    "reconstruction_reason": f"error fetching proposals: {e}",
                }
            )
            continue

        if not proposals:
            print("  No proposals found.")
            continue

        print(f"  Proposals fetched: {len(proposals)}")

        prop_rows, strat_summary = compute_proposal_metrics_for_space(
            space=space,
            proposals=proposals,
            etherscan_key=etherscan_key,
            page_size=args.page_size,
            sleep_s=args.sleep,
            max_pages=args.max_pages,
        )
        all_proposal_rows.extend(prop_rows)
        strategy_rows.append(strat_summary[space])

    if not all_proposal_rows:
        print("No proposal metrics computed; exiting.")
        return

    proposal_df = pd.DataFrame(all_proposal_rows)
    proposal_csv = os.path.join(args.out_dir, "proposal_participation_metrics.csv")
    proposal_df.to_csv(proposal_csv, index=False)
    print(f"Proposal-level metrics written to: {proposal_csv}")

    dao_df = aggregate_dao_metrics(proposal_df)
    dao_csv = os.path.join(args.out_dir, "dao_participation_metrics.csv")
    dao_df.to_csv(dao_csv, index=False)
    print(f"DAO-level metrics written to: {dao_csv}")

    strategy_df = pd.DataFrame(strategy_rows)
    strat_csv = os.path.join(args.out_dir, "strategy_summary.csv")
    strategy_df.to_csv(strat_csv, index=False)
    print(f"Strategy summary written to: {strat_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

