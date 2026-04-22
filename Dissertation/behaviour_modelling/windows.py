#!/usr/bin/env python3
"""
Sliding windows for behaviour modelling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np


@dataclass
class Window:
    window_texts: List[str]
    window_features: List[np.ndarray]
    target_label: int
    voter_id: str
    dao_cluster: int
    voter_cluster: int


def _time_feats(ts: pd.Timestamp) -> List[float]:
    if pd.isna(ts):
        return [0.0, 0.0, 0.0, 0.0]
    return [
        ts.hour / 23.0,
        ts.weekday() / 6.0,
        (ts.month - 1) / 11.0,
        (ts.day - 1) / 30.0,
    ]


def build_windows(
    df: pd.DataFrame,
    window_size: int,
    numeric_cols: List[str],
) -> List[Window]:
    df = df[df["label_id"].isin([0, 1, 2])].copy()
    out: List[Window] = []

    for voter_id, g in df.groupby("voter"):
        g = g.sort_values("vote_ts").reset_index(drop=True)
        if len(g) < window_size:
            continue

        for t in range(window_size - 1, len(g)):
            hist = list(range(t - window_size + 1, t))
            cur = t
            indices = hist + [cur]

            texts: List[str] = []
            feats: List[np.ndarray] = []
            for idx in indices:
                row = g.loc[idx]
                if idx == cur:
                    texts.append("[PREDICT] " + str(row["text"]))
                else:
                    texts.append(f"[LABEL_{int(row['label_id'])}] " + str(row["text"]))

                vec = []
                for c in numeric_cols:
                    val = row.get(c, 0.0)
                    if isinstance(val, (bool, np.bool_)):
                        vec.append(float(val))
                    else:
                        try:
                            vec.append(float(val))
                        except Exception:
                            vec.append(0.0)
                vec.extend(_time_feats(pd.to_datetime(row["vote_ts"], utc=True, errors="coerce")))
                feats.append(np.asarray(vec, dtype=np.float32))

            target = int(g.loc[cur, "label_id"])
            out.append(
                Window(
                    window_texts=texts,
                    window_features=feats,
                    target_label=target,
                    voter_id=str(voter_id),
                    dao_cluster=int(g.loc[cur, "dao_cluster"]),
                    voter_cluster=int(g.loc[cur, "voter_cluster"]),
                )
            )
    return out
