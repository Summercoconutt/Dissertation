#!/usr/bin/env python3
"""
Data loading utilities for behaviour modelling.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["voter"] = out["voter"].astype(str)
    out["vote_ts"] = pd.to_datetime(out["vote_ts"], utc=True, errors="coerce")
    out["label_id"] = pd.to_numeric(out["label_id"], errors="coerce").astype("Int64")

    for c in ["voting_power", "vp_share"]:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        out[c] = out[c].fillna(out[c].median())

    out["is_whale"] = out.get("is_whale", False).astype(bool)
    out["aligned_with_majority"] = out.get("aligned_with_majority", False).astype(bool)
    out["dao_cluster"] = pd.to_numeric(out.get("dao_cluster", -1), errors="coerce").fillna(-1).astype(int)
    out["voter_cluster"] = pd.to_numeric(out.get("voter_cluster", -1), errors="coerce").fillna(-1).astype(int)
    out["text"] = out.get("text", "").fillna("").astype(str)
    return out


def select_numeric_columns() -> List[str]:
    return ["voting_power", "vp_share", "is_whale", "aligned_with_majority", "dao_cluster", "voter_cluster"]


def split_by_voter(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    voters = df["voter"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(voters)
    n_train = int(len(voters) * train_frac)
    train_voters = set(voters[:n_train])
    train_df = df[df["voter"].isin(train_voters)].copy()
    valid_df = df[~df["voter"].isin(train_voters)].copy()
    return train_df, valid_df
