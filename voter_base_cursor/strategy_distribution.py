"""
strategy_distribution.py

Fetch strategy configuration for 441 DAO spaces and output strategy distribution.
Reads space IDs from selected_spaces_follower_knee.json.

Outputs:
  - strategy_distribution.csv: strategy_type, n_spaces, pct
  - space_strategies.csv: space, strategies_str, n_strategies, strategy_class
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests


SNAPSHOT_GRAPHQL = "https://hub.snapshot.org/graphql"


def snapshot_graphql(query: str, variables: Optional[dict] = None, timeout_s: int = 60) -> dict:
    """Call Snapshot GraphQL with retry on 5xx/429/timeouts."""
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
                raise RuntimeError(f"Retryable HTTP {r.status_code}")
            if r.status_code >= 400:
                raise RuntimeError(f"GraphQL HTTP {r.status_code}")
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL errors: {data['errors']}")
            return data["data"]
        except Exception as e:
            last_err = e
            if attempt >= max_retries - 1:
                break
            time.sleep(backoff_base * (2**attempt))
    raise RuntimeError(f"Snapshot GraphQL failed: {last_err}") from last_err


def fetch_space_strategies(space: str) -> Tuple[List[str], str]:
    """Fetch strategy names for a space via 1 proposal. Returns (strategy_names, strategy_class)."""
    q = """
    query Proposals($space: String!, $first: Int!) {
      proposals(
        first: $first
        where: { space_in: [$space] }
        orderBy: "created"
        orderDirection: desc
      ) {
        strategies { name network params }
      }
    }
    """
    data = snapshot_graphql(q, {"space": space, "first": 1})
    proposals = data.get("proposals") or []
    if not proposals:
        return [], "no_proposals"
    strats = proposals[0].get("strategies") or []
    names = [str(s.get("name") or "").strip() for s in strats if s.get("name")]
    sclass = classify_strategy_set(names)
    return names, sclass


def classify_strategy_set(names: List[str]) -> str:
    """Classify strategy set into a high-level category."""
    if not names:
        return "empty"
    joined = " ".join(names).lower()
    if len(names) == 1:
        n = names[0]
        if n in {"erc20-balance-of", "erc20-with-balance"}:
            return "simple_erc20"
        if "delegation" in n or "erc20-votes" in n or "comp-snapshot" in n:
            return "delegation"
        if any(k in n for k in ["staking", "wrapper", "lp", "uni", "balancer", "nft"]):
            return "wrapper_lp_nft"
        return "other_single"
    if any(k in joined for k in ["delegation", "erc20-votes", "comp-snapshot"]):
        return "multi_delegation"
    if any(k in joined for k in ["staking", "wrapper", "lp", "uniswap", "balancer", "nft"]):
        return "multi_wrapper_lp"
    return "multi_other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Get strategy distribution for 441 DAOs.")
    parser.add_argument(
        "--spaces_file",
        default="C:/Users/DELL/Desktop/111111/Dissertation/data_expansion/data/processed/selected_spaces_follower_knee.json",
        help="JSON file with selected_space_ids",
    )
    parser.add_argument(
        "--out_dir",
        default="out_strategy_distribution",
        help="Output directory",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep between API calls (seconds)",
    )
    args = parser.parse_args()

    with open(args.spaces_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    spaces = data.get("selected_space_ids") or data.get("spaces") or []
    if not spaces:
        raise RuntimeError("No selected_space_ids in file")

    os.makedirs(args.out_dir, exist_ok=True)

    strategy_counter: Counter = Counter()
    class_counter: Counter = Counter()
    space_rows: List[dict] = []

    n = len(spaces)
    for i, space in enumerate(spaces):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{n}] {space} ...")
        try:
            names, sclass = fetch_space_strategies(space)
        except Exception as e:
            print(f"  ERROR {space}: {e}")
            space_rows.append({
                "space": space,
                "strategies_str": "",
                "n_strategies": 0,
                "strategy_class": "fetch_error",
            })
            strategy_counter["(fetch_error)"] += 1
            class_counter["fetch_error"] += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue

        strategies_str = "|".join(sorted(set(names))) if names else "(no strategies)"
        for name in set(names):
            strategy_counter[name] += 1
        class_counter[sclass] += 1

        space_rows.append({
            "space": space,
            "strategies_str": strategies_str,
            "n_strategies": len(names),
            "strategy_class": sclass,
        })
        if args.sleep > 0:
            time.sleep(args.sleep)

    # Strategy distribution
    dist_rows = [
        {"strategy_type": k, "n_spaces": v, "pct": round(100.0 * v / n, 2)}
        for k, v in strategy_counter.most_common()
    ]
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(args.out_dir, "strategy_distribution.csv"),
        index=False,
    )
    print(f"strategy_distribution.csv: {len(dist_rows)} strategy types")

    # Strategy class distribution
    class_rows = [
        {"strategy_class": k, "n_spaces": v, "pct": round(100.0 * v / n, 2)}
        for k, v in class_counter.most_common()
    ]
    pd.DataFrame(class_rows).to_csv(
        os.path.join(args.out_dir, "strategy_class_distribution.csv"),
        index=False,
    )
    print(f"strategy_class_distribution.csv: {len(class_rows)} classes")

    # Per-space detail
    pd.DataFrame(space_rows).to_csv(
        os.path.join(args.out_dir, "space_strategies.csv"),
        index=False,
    )
    print(f"space_strategies.csv: {len(space_rows)} spaces")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
