"""Utility script: fetch latest proposal from Uniswap governance Snapshot space.

This is just for quick testing of the voter sanity-check script.
"""

from __future__ import annotations

import json

import requests

SNAPSHOT_GRAPHQL = "https://hub.snapshot.org/graphql"


def main() -> None:
    q = """
    query {
      proposals(
        first: 1
        where: { space_in: ["uniswapgovernance.eth"] }
        orderBy: "created"
        orderDirection: desc
      ) {
        id
        title
        snapshot
        space { id }
      }
    }
    """

    r = requests.post(SNAPSHOT_GRAPHQL, json={"query": q}, timeout=60)
    r.raise_for_status()
    data = r.json()
    proposal = data["data"]["proposals"][0]
    print(json.dumps(proposal, indent=2))


if __name__ == "__main__":
    main()

