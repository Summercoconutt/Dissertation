#!/usr/bin/env python3
"""
Build behaviour modelling dataset from voter clustering outputs.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

LABEL_MAP = {"for": 0, "against": 1, "abstain": 2}


def _normalise_bool(col: pd.Series) -> pd.Series:
    def to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in {"1", "true", "yes", "y", "t"}
        return bool(x)

    return col.apply(to_bool)


def build_behaviour_dataset(
    master_votes_with_cluster_parquet: Path,
    voter_cluster_assignments_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    votes = pd.read_parquet(master_votes_with_cluster_parquet)
    assign = pd.read_csv(voter_cluster_assignments_csv)

    # Prefer KMeans rows if available.
    if "clustering_method" in assign.columns:
        kmeans = assign[assign["clustering_method"] == "kmeans"].copy()
        if not kmeans.empty:
            assign = kmeans

    keep_assign_cols = ["voter", "space", "voter_cluster"]
    missing = [c for c in keep_assign_cols if c not in assign.columns]
    if missing:
        raise ValueError(f"Missing columns in assignments file: {missing}")
    assign = assign[keep_assign_cols].drop_duplicates(subset=["voter", "space"])

    df = votes.merge(assign, on=["voter", "space"], how="left")
    df["voter_cluster"] = pd.to_numeric(df["voter_cluster"], errors="coerce").fillna(-1).astype(int)
    df["dao_cluster"] = pd.to_numeric(df["dao_cluster"], errors="coerce").fillna(-1).astype(int)

    df["choice_norm"] = df["choice_norm"].astype(str).str.lower().str.strip()
    df["label_id"] = df["choice_norm"].map(LABEL_MAP).astype("Int64")
    df = df[df["label_id"].isin([0, 1, 2])].copy()

    df["vote_ts"] = pd.to_datetime(df["vote_timestamp"], utc=True, errors="coerce")
    df["text"] = df.get("proposal_title", "").fillna("").astype(str)
    df["voting_power"] = pd.to_numeric(df.get("voting_power", np.nan), errors="coerce")
    df["vp_ratio_pct"] = pd.to_numeric(df.get("vp_ratio_pct", np.nan), errors="coerce")
    df["vp_share"] = df["vp_ratio_pct"] / 100.0
    df["is_whale"] = _normalise_bool(df.get("is_whale", False))
    df["aligned_with_majority"] = _normalise_bool(df.get("aligned_with_majority", False))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
