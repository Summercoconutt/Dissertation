"""Aggregate master votes to (voter, space) features + DAO cluster merge."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import MIN_VOTES_PER_VOTER_SPACE, log


def merge_dao_clusters(votes: pd.DataFrame, assignments_csv: Path) -> pd.DataFrame:
    lab = pd.read_csv(assignments_csv)
    if "dao_cluster" not in lab.columns and "cluster" in lab.columns:
        lab = lab.rename(columns={"cluster": "dao_cluster"})
    lab = lab[["space", "dao_cluster"]].drop_duplicates(subset=["space"])
    return votes.merge(lab, on="space", how="left")


def _entropy_choice(s: pd.Series) -> float:
    u = s.value_counts()
    if u.empty:
        return 0.0
    p = u.astype(float) / u.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def build_voter_space_features(
    master_parquet: Path,
    assignments_csv: Path,
    out_with_cluster_parquet: Path,
    out_features_csv: Path,
    out_features_filtered_csv: Path,
    min_votes: int = MIN_VOTES_PER_VOTER_SPACE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    votes = pd.read_parquet(master_parquet)
    if votes.empty:
        raise ValueError("Master votes table is empty.")
    votes["vote_timestamp"] = pd.to_datetime(votes["vote_timestamp"], errors="coerce")
    votes["voting_power"] = pd.to_numeric(votes["voting_power"], errors="coerce")
    votes = merge_dao_clusters(votes, assignments_csv)
    votes.to_parquet(out_with_cluster_parquet, index=False)

    v = votes.copy()
    v["cn"] = v["choice_norm"].astype(str).str.lower()
    v["f"] = (v["cn"] == "for").astype(float)
    v["a"] = (v["cn"] == "against").astype(float)
    v["b"] = (v["cn"] == "abstain").astype(float)
    gcols = ["voter", "space"]
    G = v.groupby(gcols, dropna=False)

    feat = G.size().rename("total_votes").reset_index()
    feat = feat.merge(G["voting_power"].mean().rename("avg_voting_power").reset_index(), on=gcols)
    feat = feat.merge(G["voting_power"].std(ddof=1).rename("std_voting_power").reset_index(), on=gcols)
    feat["std_voting_power"] = feat["std_voting_power"].fillna(0.0)
    feat = feat.merge(G["f"].mean().rename("pct_for_votes").reset_index(), on=gcols)
    feat = feat.merge(G["a"].mean().rename("pct_against_votes").reset_index(), on=gcols)
    feat = feat.merge(G["b"].mean().rename("pct_abstain_votes").reset_index(), on=gcols)
    am = v.assign(_am=pd.to_numeric(v["aligned_with_majority"], errors="coerce")).groupby(gcols)["_am"].mean()
    feat = feat.merge(am.rename("pct_aligned_with_majority").reset_index(), on=gcols)
    iw = v.assign(_iw=pd.to_numeric(v["is_whale"], errors="coerce")).groupby(gcols)["_iw"].mean()
    feat = feat.merge(iw.rename("is_whale_ratio").reset_index(), on=gcols)
    feat = feat.merge(G["dao_cluster"].first().reset_index(), on=gcols)
    t0 = G["vote_timestamp"].min().reset_index(name="t0")
    t1 = G["vote_timestamp"].max().reset_index(name="t1")
    feat = feat.merge(t0, on=gcols).merge(t1, on=gcols)
    feat["active_span_days"] = ((feat["t1"] - feat["t0"]).dt.days.fillna(0).astype(int) + 1).clip(lower=1)
    feat["vote_frequency"] = feat["total_votes"] / feat["active_span_days"]
    feat = feat.merge(v.groupby(gcols)["cn"].apply(_entropy_choice).rename("vote_entropy").reset_index(), on=gcols)
    np_ = (
        v.dropna(subset=["proposal_id"])
        .groupby("space")["proposal_id"]
        .nunique()
        .rename("n_proposals_space")
        .reset_index()
    )
    feat = feat.merge(np_, on="space", how="left")
    feat["participation_rate"] = feat["total_votes"] / feat["n_proposals_space"].replace(0, np.nan)
    feat = feat.merge(v.groupby("voter")["space"].nunique().rename("n_daos_participated").reset_index(), on="voter")
    feat["log_total_votes"] = np.log1p(feat["total_votes"].astype(float))
    feat.drop(columns=["t0", "t1"], inplace=True, errors="ignore")

    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_features_csv, index=False)
    filt = feat[feat["total_votes"] >= min_votes].copy()
    filt.to_csv(out_features_filtered_csv, index=False)
    log(f"voter_space_features: {len(feat)} rows; filtered: {len(filt)} rows")
    return feat, filt
