#!/usr/bin/env python3
"""
Sliding window module: build per-voter windows of length W in time order.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class Window:
    window_texts: List[str]          # list of W texts: [LABEL_*] for history, [PREDICT] for current
    window_features: List[np.ndarray]# list of W numeric feature vectors (np.float32)
    target_label: int                # int in {0:For, 1:Against, 2:Abstain}
    voter_id: str
    cluster_id: int
    window_size: int

def _normalise_time_features(ts: pd.Timestamp) -> np.ndarray:
    # ts is timezone-aware UTC
    hour = ts.hour/23.0
    wday = ts.weekday()/6.0
    month = (ts.month-1)/11.0
    day = (ts.day-1)/30.0
    return np.array([hour, wday, month, day], dtype=np.float32)

def build_windows(df: pd.DataFrame,
                  window_size: int,
                  text_col: str,
                  label_col: str,
                  voter_col: str,
                  time_col: str,
                  cluster_col: str,
                  numeric_cols: List[str]) -> List[Window]:
    """
    Build windows by grouping per voter, sorting by time, then sliding windows of length W.
    Only rows with labels in {FOR, AGAINST, ABSTAIN} (mapped to {0,1,2}) are kept.
    """
    # Filter valid labels
    df = df.copy()
    df = df[df[label_col].isin([0,1,2])]
    # group by voter
    out: List[Window] = []
    for voter_id, g in df.groupby(voter_col):
        g = g.sort_values(time_col).reset_index(drop=True)
        if len(g) < window_size:
            continue
        # cluster id: take mode or last known in window
        cluster_vals = g[cluster_col].astype(int).tolist() if cluster_col in g.columns else [0]*len(g)
        # slide
        for t in range(window_size-1, len(g)):
            # history indices [t-W+1 .. t-1], current t
            hist_idx = list(range(t-window_size+1, t))
            cur_idx = t
            # texts: history with [LABEL_i] prefix, current with [PREDICT]
            texts = []
            for idx in hist_idx:
                lab = int(g.loc[idx, label_col])
                prefix = f"[LABEL_{lab}] "
                texts.append(prefix + str(g.loc[idx, text_col]))
            texts.append("[PREDICT] " + str(g.loc[cur_idx, text_col]))
            # numeric features per step (aligned to each step)
            feats = []
            for idx in hist_idx+[cur_idx]:
                row = g.loc[idx]
                vec = []
                # basic numeric cols (already numeric)
                for col in numeric_cols:
                    v = row.get(col, 0.0)
                    try:
                        v = float(v)
                    except Exception:
                        v = 0.0
                    vec.append(v)
                # boolean flags (Is Whale, Aligned With Majority) as 0/1 if present
                for bcol in ["Is Whale","Aligned With Majority"]:
                    if bcol in g.columns:
                        val = row.get(bcol, False)
                        if isinstance(val, str):
                            val = val.strip().lower() in ("1","true","yes")
                        vec.append(1.0 if bool(val) else 0.0)
                # time features (normalised)
                ts = pd.to_datetime(row[time_col], utc=True, errors="coerce")
                if pd.isna(ts):
                    feats_time = np.zeros(4, dtype=np.float32)
                else:
                    feats_time = _normalise_time_features(ts)
                feats.extend([vec + feats_time.tolist()])
            # target label at step t
            y = int(g.loc[cur_idx, label_col])
            # cluster id for the window: majority in the window (fallback to last)
            win_clusters = [cluster_vals[idx] for idx in hist_idx+[cur_idx]]
            if len(win_clusters)>0:
                # mode
                vals, counts = np.unique(win_clusters, return_counts=True)
                cluster_id = int(vals[counts.argmax()])
            else:
                cluster_id = int(cluster_vals[cur_idx])
            # record
            feats = [np.asarray(f, dtype=np.float32) for f in feats]
            out.append(Window(
                window_texts=texts,
                window_features=feats,
                target_label=y,
                voter_id=str(voter_id),
                cluster_id=cluster_id,
                window_size=window_size
            ))
    return out
