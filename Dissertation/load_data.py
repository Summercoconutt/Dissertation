#!/usr/bin/env python3
"""
Data loading and preprocessing for Agent 2.
"""
from typing import Tuple, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np

VALID_LABELS = {"FOR":0, "AGAINST":1, "ABSTAIN":2}  # For=0, Against=1, Abstain=2

def load_and_merge(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardise key columns and types
    df = df.copy()
    # canonical voter
    if "voter" in df.columns:
        df["voter"] = df["voter"].astype(str)
    elif "Voter" in df.columns:
        df["voter"] = df["Voter"].astype(str)
    else:
        raise ValueError("Missing voter column")
    # time
    tcol = "Vote Timestamp" if "Vote Timestamp" in df.columns else "Created Time"
    df["vote_ts"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    # label
    if "Vote Label" in df.columns:
        lab = df["Vote Label"].astype(str).str.upper().str.strip()
        df["label_id"] = lab.map(VALID_LABELS).astype("Int64")
    else:
        raise ValueError("Missing 'Vote Label' column")
    # text
    text_col = "Proposal Title" if "Proposal Title" in df.columns else "Choice_Text"
    df["text"] = df[text_col].astype(str)
    # numeric basics
    if "Voting Power" in df.columns:
        df["vp"] = pd.to_numeric(df["Voting Power"], errors="coerce")
    else:
        df["vp"] = np.nan
    if "VP Ratio (%)" in df.columns:
        df["vp_share"] = pd.to_numeric(df["VP Ratio (%)"], errors="coerce")/100.0
    else:
        df["vp_share"] = np.nan
    # booleans
    def to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in ("1","true","yes","y","t")
        return bool(x)
    df["is_whale"] = df.get("Is Whale", False).apply(to_bool) if "Is Whale" in df.columns else False
    df["aligned_majority"] = df.get("Aligned With Majority", False).apply(to_bool) if "Aligned With Majority" in df.columns else False
    # cluster id provided
    if "cluster" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(0).astype(int)
    else:
        df["cluster_id"] = 0
    return df

def select_numeric_columns(df: pd.DataFrame) -> List[str]:
    # safe per-step numeric features (no future aggregation)
    cols = []
    for name in ["vp","vp_share"]:
        if name in df.columns:
            cols.append(name)
    # booleans will be appended in window builder; time features generated later
    return cols

def split_by_voter(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42):
    voters = df["voter"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(voters)
    n_train = int(len(voters)*train_frac)
    train_v = set(voters[:n_train])
    train_df = df[df["voter"].isin(train_v)].copy()
    valid_df = df[~df["voter"].isin(train_v)].copy()
    return train_df, valid_df

