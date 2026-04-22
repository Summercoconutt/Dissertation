from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def parse_space_dir_name(name: str) -> Optional[str]:
    if name.startswith("space="):
        return name.split("space=", 1)[1].strip()
    return name.strip() if name else None


def parse_proposal_file_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("proposal="):
        return stem.split("proposal=", 1)[1]
    return stem


def iter_snapshot_files(base_spaces_dir: Path, selected_spaces: Optional[Sequence[str]] = None) -> Iterable[Tuple[str, Path]]:
    selected_set = set(selected_spaces) if selected_spaces else None
    for space_dir in sorted(base_spaces_dir.iterdir()):
        if not space_dir.is_dir():
            continue
        space = parse_space_dir_name(space_dir.name)
        if not space:
            continue
        if selected_set is not None and space not in selected_set:
            continue

        for p in sorted(space_dir.glob("proposal=*.parquet")):
            yield space, p
        for p in sorted(space_dir.glob("proposal=*.csv")):
            yield space, p


def normalize_vote_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "Voter": "voter",
        "Voting Power": "voting_power",
        "Vote Timestamp": "timestamp",
        "Created Time": "created_time",
        "Choice": "choice",
        "Vote Label": "vote_label",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    if "choice" not in out.columns:
        out["choice"] = np.nan
    if "vote_label" not in out.columns:
        out["vote_label"] = np.nan
    return out


def normalize_choice(choice: object, vote_label: object) -> str:
    cnum = pd.to_numeric(pd.Series([choice]), errors="coerce").iloc[0]
    if pd.notna(cnum):
        ci = int(cnum)
        if ci == 1:
            return "for"
        if ci == 2:
            return "against"
        if ci == 3:
            return "abstain"

    if pd.notna(vote_label):
        txt = str(vote_label).strip().lower().replace("_", " ").replace("-", " ")
        if "against" in txt:
            return "against"
        if "abstain" in txt:
            return "abstain"
        if "for" in txt or txt in {"yes", "yay", "approve", "support"}:
            return "for"

    return "unknown"


def safe_gini(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().astype(float).values
    if arr.size == 0:
        return np.nan
    if np.allclose(arr, 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2 * np.sum((np.arange(1, n + 1) * arr)) / (n * arr.sum())) - (n + 1) / n)


def safe_hhi(shares: pd.Series) -> float:
    s = pd.to_numeric(shares, errors="coerce").dropna().astype(float)
    if s.empty:
        return np.nan
    return float(np.square(s).sum())

