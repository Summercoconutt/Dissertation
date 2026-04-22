from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from utils import iter_snapshot_files, log, normalize_choice, normalize_vote_columns, parse_proposal_file_name, warn


def load_votes(base_spaces_dir: Path, selected_spaces: Optional[Sequence[str]] = None) -> pd.DataFrame:
    frames = []
    n_files = 0
    for space, path in iter_snapshot_files(base_spaces_dir, selected_spaces=selected_spaces):
        n_files += 1
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            df = normalize_vote_columns(df)
            df["space"] = space
            df["proposal_id"] = parse_proposal_file_name(path.name)
            frames.append(df)
        except Exception as exc:
            warn(f"Failed reading {path}: {exc}")

    if n_files == 0 or not frames:
        raise RuntimeError("No input files loaded from spaces directory.")

    merged = pd.concat(frames, ignore_index=True)
    merged["voter"] = merged.get("voter", pd.Series(dtype="object")).astype("string")
    merged["voting_power"] = pd.to_numeric(merged.get("voting_power", np.nan), errors="coerce")
    merged["timestamp"] = pd.to_datetime(merged.get("timestamp", pd.NaT), errors="coerce")
    merged["created_time"] = pd.to_datetime(merged.get("created_time", pd.NaT), errors="coerce")
    merged = merged.dropna(subset=["voter", "voting_power"])
    merged["choice_norm"] = merged.apply(lambda r: normalize_choice(r.get("choice"), r.get("vote_label")), axis=1)
    # Arrow-safe fix: mixed object columns (especially choice) can break parquet serialization.
    if "choice" in merged.columns:
        merged["choice"] = merged["choice"].apply(lambda x: np.nan if pd.isna(x) else str(x))
    for col in merged.columns:
        if merged[col].dtype == "object":
            sample = merged[col].dropna().head(200)
            if not sample.empty and sample.map(lambda x: isinstance(x, (dict, list, tuple, set))).any():
                merged[col] = merged[col].apply(lambda x: np.nan if pd.isna(x) else str(x))
    log(f"Loaded rows: {len(merged):,} from files: {n_files:,}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-spaces-dir",
        default=r"C:\Users\DELL\Desktop\snapshot_votes_data\snapshot_votes_441\spaces",
    )
    parser.add_argument("--spaces", default="", help="Comma-separated spaces subset")
    parser.add_argument("--out", default=r"data\raw\snapshot_votes.parquet")
    args = parser.parse_args()

    selected = [s.strip() for s in args.spaces.split(",") if s.strip()] or None
    df = load_votes(Path(args.base_spaces_dir), selected_spaces=selected)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log(f"Saved raw merged votes: {out}")


if __name__ == "__main__":
    main()

