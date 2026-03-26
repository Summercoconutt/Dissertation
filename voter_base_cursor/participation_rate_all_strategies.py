"""participation_rate_all_strategies.py

Participation metrics pipeline for Snapshot spaces, supporting:

- A universal **voting-power participation rate** (vp_participation_rate)
  for all proposals, based solely on Snapshot's own voting power outputs.
- An optional **address-based participation rate** (address_participation_rate)
  only for simple ERC20 balance strategies on Ethereum mainnet.

Methodology (summary)
---------------------
For each proposal:
- vp_used  = sum(vote.vp) over all Snapshot votes (including Abstain, etc.)
- vp_total = proposal.scores_total
- vp_participation_rate = vp_used / vp_total  (if vp_total > 0, else null)

For address-based metrics (simple ERC20 balance on mainnet only):
- actual voters   = unique addresses from Snapshot votes
- eligible voters = unique addresses with positive ERC20 balance at snapshot
                    (reconstructed via Etherscan Transfer logs)
- address_participation_rate = n_actual_voters / n_eligible_voters

For all other strategies:
- vp_participation_rate is still computed.
- Address-based metrics are left null and clearly flagged as unsupported.

Environment variables
---------------------
- ETHERSCAN_API_KEY : required ONLY for address-based reconstruction.
- RPC_URL           : optional (logged but not used in v1).
"""

from __future__ import annotations

import argparse
import json
import os
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
    """Fetch up to `first` most recent proposals for a space, including scores & strategies."""
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
        type
        snapshot
        created
        choices
        scores
        scores_total
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


def get_unique_voters_and_vp(votes: List[dict]) -> Tuple[Set[str], float]:
    """Return (unique_voters_set, vp_used_sum)."""
    voters: Set[str] = set()
    vp_used = 0.0
    for v in votes:
        addr = (v.get("voter") or "").strip().lower()
        if addr:
            voters.add(addr)
        try:
            vp_val = float(v.get("vp") or 0.0)
        except (TypeError, ValueError):
            vp_val = 0.0
        vp_used += vp_val
    return voters, vp_used


# =========================
# Strategy classification
# =========================


def classify_strategy_set(strategies: List[dict]) -> Tuple[str, str, bool, str, Optional[str]]:
    """Classify a list of strategies for a space/proposal.

    Returns:
        detected_strategies (comma-separated names),
        strategy_class (see below),
        address_supported (bool),
        reconstruction_reason (str),
        token_address_if_simple (Optional[str])

    strategy_class values:
        - simple_erc20_balance_mainnet
        - complex_delegation
        - complex_staking
        - complex_wrapper
        - complex_multi_strategy
        - unsupported_unknown
    """
    if not strategies:
        return ("none", "unsupported_unknown", False, "no strategies returned from Snapshot", None)

    names = [str(s.get("name") or "").strip() for s in strategies]
    detected = ",".join(sorted(set(names)))

    # Heuristic classification
    if len(strategies) == 1:
        s0 = strategies[0]
        name0 = names[0]
        params0 = s0.get("params") or {}
        network0 = str(s0.get("network") or "1")

        # Simple ERC20 balance-based on Ethereum mainnet
        if name0 in {"erc20-balance-of", "erc20-with-balance"}:
            token_addr = params0.get("address") or params0.get("token")
            if network0 != "1":
                return (
                    detected,
                    "unsupported_unknown",
                    False,
                    f"ERC20 balance strategy but network={network0} != 1",
                    None,
                )
            if not token_addr:
                return (
                    detected,
                    "unsupported_unknown",
                    False,
                    "ERC20 balance strategy but missing token address in params",
                    None,
                )
            return (
                detected,
                "simple_erc20_balance_mainnet",
                True,
                "address-based eligibility supported for simple ERC20 balance on mainnet",
                str(token_addr),
            )

        # Delegation-style strategies (non-simple)
        if "delegation" in name0 or "erc20-votes" in name0 or "comp-snapshot" in name0:
            return (
                detected,
                "complex_delegation",
                False,
                "delegation / votes-style strategy; address-based eligibility not reconstructed in v1",
                None,
            )

        # Staking / wrapper / LP / NFT
        if any(
            key in name0
            for key in ["staking", "wrapper", "lp", "uniswap", "balancer", "sushiswap", "uni-v3", "nft"]
        ):
            return (
                detected,
                "complex_wrapper",
                False,
                "staking / wrapper / LP / NFT-like strategy; address-based eligibility not reconstructed in v1",
                None,
            )

        # Fallback: unknown single strategy
        return (
            detected,
            "unsupported_unknown",
            False,
            f"single strategy {name0!r} not recognized as simple ERC20; address-based eligibility unsupported",
            None,
        )

    # Multi-strategy
    # Try to detect if any of the names strongly indicate delegation/staking/wrappers
    joined = " ".join(names).lower()
    if any(key in joined for key in ["delegation", "erc20-votes", "comp-snapshot"]):
        sclass = "complex_delegation"
        reason = "multi-strategy configuration including delegation/votes-like components"
    elif any(key in joined for key in ["staking", "wrapper", "lp", "uniswap", "balancer", "sushiswap", "uni-v3", "nft"]):
        sclass = "complex_wrapper"
        reason = "multi-strategy configuration including staking/wrapper/LP/NFT components"
    else:
        sclass = "complex_multi_strategy"
        reason = "multi-strategy configuration; simple ERC20 eligibility not guaranteed"

    return (detected, sclass, False, reason, None)


# =========================
# Etherscan helpers (for simple ERC20 only)
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
# Metric computation
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

    # Assume strategies stable across proposals; use first proposal's strategies for classification.
    all_strats = proposals[0].get("strategies") if proposals else []
    detected_strats, strategy_class, addr_supported, reason, token_addr = classify_strategy_set(all_strats)

    strategy_summary[space] = {
        "space": space,
        "detected_strategies": detected_strats,
        "strategy_class": strategy_class,
        "address_reconstruction_supported": addr_supported,
        "reconstruction_reason": reason,
        "token_address_if_simple": token_addr or "",
    }

    for p in proposals:
        pid = p["id"]
        snapshot_block = int(p["snapshot"])
        proposal_type = p.get("type") or ""
        choices = p.get("choices") or []
        scores_total_raw = p.get("scores_total", 0)
        try:
            vp_total = float(scores_total_raw or 0.0)
        except (TypeError, ValueError):
            vp_total = 0.0

        votes = fetch_all_votes(pid)
        voters, vp_used = get_unique_voters_and_vp(votes)

        if vp_total > 0:
            vp_pr = vp_used / vp_total
        else:
            vp_pr = None

        # Base row with VP metrics (always filled)
        row = {
            "space": space,
            "proposal_id": pid,
            "snapshot_block": snapshot_block,
            "proposal_type": proposal_type,
            "choices": "|".join(str(c) for c in choices),
            "strategy_class": strategy_class,
            "vp_used": vp_used,
            "vp_total": vp_total,
            "vp_participation_rate": vp_pr,
            "n_actual_voters": len(voters),
            "n_eligible_voters": None,
            "address_participation_rate": None,
            "address_support_status": "unsupported",
            "address_support_reason": reason,
        }

        # Optional address-based metrics for simple ERC20 mainnet only.
        if strategy_class == "simple_erc20_balance_mainnet" and addr_supported:
            if not etherscan_key:
                row["address_support_status"] = "unsupported"
                row["address_support_reason"] = "ETHERSCAN_API_KEY missing"
            else:
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
                        row["address_participation_rate"] = len(voters) / n_eligible
                        row["address_support_status"] = "supported"
                        row["address_support_reason"] = (
                            "simple ERC20 eligibility reconstructed via Etherscan transfers"
                        )
                    else:
                        row["address_support_status"] = "unsupported"
                        row["address_support_reason"] = (
                            "no positive-balance holders reconstructed (0 eligible voters)"
                        )
                except Exception as e:  # noqa: BLE001
                    row["address_support_status"] = "unsupported"
                    row["address_support_reason"] = f"error during eligibility reconstruction: {e}"

        proposal_rows.append(row)

    return proposal_rows, strategy_summary


def aggregate_dao_metrics(proposal_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate proposal-level metrics to DAO/space-level metrics."""
    rows: List[dict] = []
    for space, g in proposal_df.groupby("space"):
        n_proposals = int(len(g))

        # VP metrics: use all proposals with vp_participation_rate not null.
        vp = g["vp_participation_rate"].dropna()
        if not vp.empty:
            mean_vp = float(vp.mean())
            median_vp = float(vp.median())
            # Weighted mean by vp_total (if available and >0)
            mask = g["vp_participation_rate"].notna() & (g["vp_total"] > 0)
            if mask.any():
                numer = float((g.loc[mask, "vp_participation_rate"] * g.loc[mask, "vp_total"]).sum())
                denom = float(g.loc[mask, "vp_total"].sum())
                w_mean_vp = numer / denom if denom > 0 else float("nan")
            else:
                w_mean_vp = float("nan")
        else:
            mean_vp = float("nan")
            median_vp = float("nan")
            w_mean_vp = float("nan")

        # Address-based metrics: only supported proposals with non-null address_participation_rate.
        addr_supported_mask = g["address_support_status"] == "supported"
        addr_supported_vals = g.loc[addr_supported_mask, "address_participation_rate"].dropna()
        if not addr_supported_vals.empty:
            mean_addr = float(addr_supported_vals.mean())
        else:
            mean_addr = float("nan")

        n_address_supported = int(addr_supported_mask.sum())
        n_address_unsupported = int(n_proposals - n_address_supported)
        address_support_ratio = (
            n_address_supported / n_proposals if n_proposals > 0 else 0.0
        )

        rows.append(
            {
                "space": space,
                "n_proposals": n_proposals,
                "mean_vp_participation_rate": mean_vp,
                "median_vp_participation_rate": median_vp,
                "weighted_mean_vp_participation_rate": w_mean_vp,
                "mean_address_participation_rate": mean_addr,
                "n_address_supported": n_address_supported,
                "n_address_unsupported": n_address_unsupported,
                "address_support_ratio": address_support_ratio,
            }
        )

    return pd.DataFrame(rows)


# =========================
# CLI
# =========================


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Participation metrics for Snapshot spaces (vp + optional address-based)."
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        help="Snapshot space ids (e.g. aavedao.eth uniswapgovernance.eth)",
    )
    parser.add_argument(
        "--spaces_file",
        help="JSON file with 'selected_space_ids' array (overrides --spaces if given)",
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
        default="out_participation_all",
        help="Output directory for CSVs (default: out_participation_all).",
    )
    args = parser.parse_args(argv)

    if args.spaces_file:
        with open(args.spaces_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        spaces = data.get("selected_space_ids") or data.get("spaces") or []
        if not spaces:
            raise RuntimeError(
                f"spaces_file {args.spaces_file!r} has no 'selected_space_ids' or 'spaces'"
            )
    elif args.spaces:
        spaces = args.spaces
    else:
        parser.error("Provide --spaces or --spaces_file")

    os.makedirs(args.out_dir, exist_ok=True)
    rpc_url = os.environ.get("RPC_URL")
    etherscan_key = os.environ.get("ETHERSCAN_API_KEY")

    print("=== Participation metrics (all strategies) ===")
    print(f"Spaces            : {len(spaces)} from {'file' if args.spaces_file else 'cli'}")
    print(f"Max proposals     : {args.max_proposals}")
    print(f"RPC_URL set       : {'yes' if rpc_url else 'no'}")
    print(f"ETHERSCAN_API_KEY : {'yes' if etherscan_key else 'no'}")
    print()

    all_proposal_rows: List[dict] = []
    strategy_rows: List[dict] = []

    for space in spaces:
        print(f"--- Space: {space} ---")
        try:
            proposals = fetch_space_proposals(space, args.max_proposals)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR fetching proposals: {e}")
            strategy_rows.append(
                {
                    "space": space,
                    "detected_strategies": "",
                    "strategy_class": "unsupported_unknown",
                    "address_reconstruction_supported": False,
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

