import argparse
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient of a 1D array."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Relative mean difference formula
    gini = (2.0 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1) / n
    return float(gini)


def compute_proposal_level(df_votes: pd.DataFrame, df_voter_base: pd.DataFrame) -> pd.DataFrame:
    """
    Build proposal-level dataset from raw vote-level data.

    Assumptions (because we only have per-space CSVs here):
    - Each input CSV corresponds to a single Snapshot space (DAO).
    - Proposal identity is given by `Proposal Title` + its `Created Time`.
    - Eligible voters per proposal are approximated by the total number of
      unique voters in `voter_base.csv` (space-level voter base).
    """
    # Parse timestamps
    df_votes = df_votes.copy()
    df_votes["Created Time"] = pd.to_datetime(df_votes["Created Time"])
    df_votes["Vote Timestamp"] = pd.to_datetime(df_votes["Vote Timestamp"])

    # Space-level "eligible voters" approximation from voter_base
    total_eligible_voters = df_voter_base["voter"].nunique()

    # Group by proposal
    group_cols = ["Proposal Title", "Created Time"]

    def aggregate_proposal(group: pd.DataFrame) -> Dict[str, Any]:
        voters = group["Voter"].nunique()
        total_vp = group["Voting Power"].sum()
        return {
            "n_voters": voters,
            "total_voting_power": total_vp,
            "eligible_voters": total_eligible_voters,
        }

    df_prop = (
        df_votes.groupby(group_cols, as_index=False)
        .apply(lambda g: pd.Series(aggregate_proposal(g)))
    )

    # Participation rate
    df_prop["participation_rate"] = df_prop["n_voters"] / df_prop["eligible_voters"].replace(
        {0: np.nan}
    )

    return df_prop


def compute_dao_metrics(
    space_name: str,
    df_votes: pd.DataFrame,
    df_voter_base: pd.DataFrame,
    df_space_strategies: pd.DataFrame,
) -> pd.DataFrame:
    """Compute DAO-level metrics for a single space."""
    # Proposal-level metrics
    df_prop = compute_proposal_level(df_votes, df_voter_base)

    # --- 1. Governance activity ---
    n_proposals = len(df_prop)

    if n_proposals > 0:
        t_min = df_prop["Created Time"].min()
        t_max = df_prop["Created Time"].max()
        # At least 1 month to avoid division by zero
        n_months = max((t_max - t_min).days / 30.0, 1.0)
        proposals_per_month = n_proposals / n_months
        active_days = df_prop["Created Time"].dt.date.nunique()
    else:
        proposals_per_month = np.nan
        active_days = 0

    # --- 2. Voter scale ---
    total_unique_voters = df_votes["Voter"].nunique()
    avg_voters_per_proposal = df_prop["n_voters"].mean() if n_proposals > 0 else np.nan
    median_voters_per_proposal = df_prop["n_voters"].median() if n_proposals > 0 else np.nan

    # --- 3. Eligible voters ---
    avg_eligible_voters = df_prop["eligible_voters"].mean() if n_proposals > 0 else np.nan
    median_eligible_voters = df_prop["eligible_voters"].median() if n_proposals > 0 else np.nan

    # --- 4. Participation ---
    if n_proposals > 0:
        pr = df_prop["participation_rate"].dropna()
        if len(pr) > 0:
            avg_participation_rate = pr.mean()
            median_participation_rate = pr.median()
            std_participation_rate = pr.std(ddof=0)
            p25_participation_rate = pr.quantile(0.25)
            p75_participation_rate = pr.quantile(0.75)
        else:
            avg_participation_rate = median_participation_rate = std_participation_rate = np.nan
            p25_participation_rate = p75_participation_rate = np.nan
    else:
        avg_participation_rate = median_participation_rate = std_participation_rate = np.nan
        p25_participation_rate = p75_participation_rate = np.nan

    # --- 5. Voter stability --- (from voter_base.csv)
    if len(df_voter_base) > 0:
        total_voters_base = df_voter_base["voter"].nunique()
        repeat_voters = df_voter_base[df_voter_base["total_votes"] > 1]["voter"].nunique()
        repeat_voter_ratio = repeat_voters / total_voters_base

        # Define "heavy voters" as those with >= 5 votes across proposals
        heavy_voters = df_voter_base[df_voter_base["total_votes"] >= 5]["voter"].nunique()
        heavy_voter_ratio = heavy_voters / total_voters_base

        avg_votes_per_voter = df_voter_base["total_votes"].mean()
    else:
        repeat_voter_ratio = heavy_voter_ratio = avg_votes_per_voter = np.nan

    # --- 6. Voter overlap --- (between proposals, Jaccard similarity)
    voter_sets: List[set] = []
    for _, g in df_votes.groupby(["Proposal Title", "Created Time"]):
        voter_sets.append(set(g["Voter"].unique()))

    overlaps: List[float] = []
    for i in range(len(voter_sets)):
        for j in range(i + 1, len(voter_sets)):
            a, b = voter_sets[i], voter_sets[j]
            union = a | b
            if len(union) == 0:
                continue
            jaccard = len(a & b) / len(union)
            overlaps.append(jaccard)

    if overlaps:
        mean_voter_overlap = float(np.mean(overlaps))
        median_voter_overlap = float(np.median(overlaps))
    else:
        mean_voter_overlap = median_voter_overlap = np.nan

    # --- 7. Voting power concentration ---
    top10_shares = []
    top5_shares = []
    top1_shares = []
    gini_list = []

    for _, g in df_votes.groupby(["Proposal Title", "Created Time"]):
        vp = g["Voting Power"].astype(float).values
        total_vp = vp.sum()
        if total_vp <= 0:
            continue
        vp_sorted = np.sort(vp)[::-1]
        top10_shares.append(vp_sorted[:10].sum() / total_vp)
        top5_shares.append(vp_sorted[:5].sum() / total_vp)
        top1_shares.append(vp_sorted[:1].sum() / total_vp)
        gini_list.append(gini_coefficient(vp))

    avg_top10_vp_share = float(np.mean(top10_shares)) if top10_shares else np.nan
    avg_top5_vp_share = float(np.mean(top5_shares)) if top5_shares else np.nan
    avg_top1_vp_share = float(np.mean(top1_shares)) if top1_shares else np.nan
    avg_gini_voting_power = float(np.mean(gini_list)) if gini_list else np.nan

    # --- 8. Governance design --- (from space_strategies.csv)
    row_strategy = df_space_strategies[df_space_strategies["space"] == space_name]
    if not row_strategy.empty:
        n_unique_strategies = float(row_strategy["n_strategies"].iloc[0])
        # With our data we only know space-level strategy config, so we assume
        # each proposal uses this configuration.
        avg_strategies_per_proposal = n_unique_strategies
        multi_strategy_ratio = 1.0 if n_unique_strategies > 1 else 0.0
    else:
        n_unique_strategies = avg_strategies_per_proposal = multi_strategy_ratio = np.nan

    data = {
        "space": space_name,
        # 1. Governance activity
        "n_proposals": n_proposals,
        "proposal_per_month": proposals_per_month,
        "active_days": active_days,
        # 2. Voter scale
        "total_unique_voters": total_unique_voters,
        "avg_voters_per_proposal": avg_voters_per_proposal,
        "median_voters_per_proposal": median_voters_per_proposal,
        # 3. Eligible voters
        "avg_eligible_voters": avg_eligible_voters,
        "median_eligible_voters": median_eligible_voters,
        # 4. Participation
        "avg_participation_rate": avg_participation_rate,
        "median_participation_rate": median_participation_rate,
        "std_participation_rate": std_participation_rate,
        "p25_participation_rate": p25_participation_rate,
        "p75_participation_rate": p75_participation_rate,
        # 5. Voter stability
        "repeat_voter_ratio": repeat_voter_ratio,
        "heavy_voter_ratio": heavy_voter_ratio,
        "avg_votes_per_voter": avg_votes_per_voter,
        # 6. Voter overlap
        "mean_voter_overlap": mean_voter_overlap,
        "median_voter_overlap": median_voter_overlap,
        # 7. Voting power concentration
        "avg_top10_vp_share": avg_top10_vp_share,
        "avg_top5_vp_share": avg_top5_vp_share,
        "avg_top1_vp_share": avg_top1_vp_share,
        "avg_gini_voting_power": avg_gini_voting_power,
        # 8. Governance design
        "n_unique_strategies": n_unique_strategies,
        "avg_strategies_per_proposal": avg_strategies_per_proposal,
        "multi_strategy_ratio": multi_strategy_ratio,
    }

    return pd.DataFrame([data])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DAO-level governance metrics for a single Snapshot space."
    )
    parser.add_argument("--space-name", required=True, help="Snapshot space name, e.g. uniswapgovernance.eth")
    parser.add_argument(
        "--votes-file",
        required=True,
        help="Path to snapshot_votes_with_timestamp_choice.csv for this space.",
    )
    parser.add_argument(
        "--voter-base-file",
        required=True,
        help="Path to voter_base.csv for this space.",
    )
    parser.add_argument(
        "--space-strategies-file",
        required=True,
        help="Path to space_strategies.csv containing strategy configs for many spaces.",
    )
    parser.add_argument(
        "--output-file",
        default="dao_metrics_raw.csv",
        help="Output CSV path for DAO-level metrics (will be created or appended).",
    )

    args = parser.parse_args()

    # Load data
    df_votes = pd.read_csv(args.votes_file)
    df_voter_base = pd.read_csv(args.voter_base_file)
    df_space_strategies = pd.read_csv(args.space_strategies_file)

    # Basic column checks with informative errors
    required_votes_cols = {
        "Proposal Title",
        "Created Time",
        "Voter",
        "Voting Power",
        "Vote Timestamp",
    }
    missing_votes = required_votes_cols - set(df_votes.columns)
    if missing_votes:
        raise ValueError(f"votes-file missing columns: {missing_votes}")

    required_voter_base_cols = {"voter", "total_votes"}
    missing_vb = required_voter_base_cols - set(df_voter_base.columns)
    if missing_vb:
        raise ValueError(f"voter-base-file missing columns: {missing_vb}")

    if "space" not in df_space_strategies.columns or "n_strategies" not in df_space_strategies.columns:
        raise ValueError("space-strategies-file must contain 'space' and 'n_strategies' columns.")

    # Compute metrics
    df_dao = compute_dao_metrics(
        space_name=args.space_name,
        df_votes=df_votes,
        df_voter_base=df_voter_base,
        df_space_strategies=df_space_strategies,
    )

    # Save / append to output CSV
    output_path = args.output_file
    if os.path.exists(output_path):
        # Append without header
        df_dao.to_csv(output_path, mode="a", index=False, header=False)
    else:
        df_dao.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

