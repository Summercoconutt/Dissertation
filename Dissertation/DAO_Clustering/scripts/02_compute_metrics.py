from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import log, safe_gini, safe_hhi


def compute_dao_metrics(votes: pd.DataFrame) -> pd.DataFrame:
    # proposal-level helpers
    proposal_stats = (
        votes.groupby(["space", "proposal_id"], as_index=False)
        .agg(
            n_voters=("voter", "nunique"),
            proposal_votes=("voter", "size"),
            sum_voting_power=("voting_power", "sum"),
            created_time=("created_time", "min"),
        )
    )

    # size & activity
    size_activity = (
        votes.groupby("space", as_index=False)
        .agg(
            n_unique_voters=("voter", "nunique"),
            total_votes=("voter", "size"),
        )
        .merge(
            proposal_stats.groupby("space", as_index=False).agg(
                n_proposals=("proposal_id", "nunique"),
                avg_voters_per_proposal=("n_voters", "mean"),
                median_voters_per_proposal=("n_voters", "median"),
            ),
            on="space",
            how="left",
        )
    )

    # participation proxies
    size_activity["voter_turnout_proxy"] = size_activity["avg_voters_per_proposal"] / size_activity[
        "n_unique_voters"
    ].replace({0: np.nan})

    # concentration metrics at proposal then DAO-average
    per_proposal_conc = []
    for (space, proposal_id), g in votes.groupby(["space", "proposal_id"]):
        total = g["voting_power"].sum()
        shares = g["voting_power"] / total if total > 0 else pd.Series(dtype=float)
        per_proposal_conc.append(
            {
                "space": space,
                "proposal_id": proposal_id,
                "gini_voting_power": safe_gini(g["voting_power"]),
                "hhi_voting_power": safe_hhi(shares),
                "whale_ratio_top1pct": float(shares[shares >= shares.quantile(0.99)].sum()) if len(shares) > 0 else np.nan,
            }
        )
    conc = pd.DataFrame(per_proposal_conc).groupby("space", as_index=False).agg(
        gini_voting_power=("gini_voting_power", "mean"),
        hhi_voting_power=("hhi_voting_power", "mean"),
        whale_ratio_top1pct=("whale_ratio_top1pct", "mean"),
    )

    # temporal dynamics
    proposal_time = proposal_stats.groupby("space", as_index=False).agg(
        first_proposal=("created_time", "min"),
        last_proposal=("created_time", "max"),
    )
    proposal_time["active_days"] = (proposal_time["last_proposal"] - proposal_time["first_proposal"]).dt.days.clip(lower=1)
    proposal_time = proposal_time.merge(
        proposal_stats.groupby("space", as_index=False).agg(n_proposals=("proposal_id", "nunique")),
        on="space",
        how="left",
    )
    proposal_time["proposal_frequency_per_30d"] = proposal_time["n_proposals"] / (proposal_time["active_days"] / 30.0)

    repeat = votes.groupby(["space", "voter"], as_index=False).size().rename(columns={"size": "votes_by_voter"})
    repeat_rate = repeat.groupby("space", as_index=False).agg(
        repeat_voter_rate=("votes_by_voter", lambda s: float((s > 1).mean()))
    )

    dao_metrics = (
        size_activity.merge(conc, on="space", how="left")
        .merge(proposal_time[["space", "proposal_frequency_per_30d"]], on="space", how="left")
        .merge(repeat_rate, on="space", how="left")
    )
    return dao_metrics, proposal_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=r"data\raw\snapshot_votes.parquet")
    parser.add_argument("--out-dao", default=r"data\interim\dao_metrics.parquet")
    parser.add_argument("--out-proposal", default=r"data\interim\proposal_metrics.parquet")
    args = parser.parse_args()

    votes = pd.read_parquet(Path(args.in_path))
    dao_metrics, proposal_metrics = compute_dao_metrics(votes)

    Path(args.out_dao).parent.mkdir(parents=True, exist_ok=True)
    dao_metrics.to_parquet(args.out_dao, index=False)
    proposal_metrics.to_parquet(args.out_proposal, index=False)
    log(f"Saved DAO metrics: {args.out_dao}")
    log(f"Saved proposal metrics: {args.out_proposal}")


if __name__ == "__main__":
    main()

