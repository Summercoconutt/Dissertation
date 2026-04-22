from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import log


def build_feature_table(
    dao_metrics: pd.DataFrame,
    participation_summary_path: Path,
    representative_path: Path,
) -> pd.DataFrame:
    out = dao_metrics.copy()

    if participation_summary_path.exists():
        part = pd.read_csv(participation_summary_path)
        keep = [c for c in part.columns if c == "space" or c.startswith("mean_") or c.startswith("std_")]
        part = part[keep]
        out = out.merge(part, on="space", how="left")
    else:
        log(f"Participation summary not found, skip merge: {participation_summary_path}")

    if representative_path.exists():
        rep = pd.read_csv(representative_path)
        out = out.merge(rep, on="space", how="left")
    else:
        log(f"Representative voter table not found, skip merge: {representative_path}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dao-metrics", default=r"data\interim\dao_metrics.parquet")
    parser.add_argument(
        "--participation-summary",
        default=r"C:\Users\DELL\Desktop\snapshot_votes_data\snapshot_votes_441\outputs\participation_rate\dao_level_participation_summary.csv",
    )
    parser.add_argument(
        "--representative-csv",
        default=r"C:\Users\DELL\Desktop\snapshot_votes_data\snapshot_votes_441\outputs\latest_dao_representative.csv",
        help="Path to dao_representative_voters_*.csv. Update this path if needed.",
    )
    parser.add_argument("--out", default=r"data\processed\dao_feature_table.parquet")
    args = parser.parse_args()

    dao_metrics = pd.read_parquet(Path(args.dao_metrics))
    feature_table = build_feature_table(
        dao_metrics=dao_metrics,
        participation_summary_path=Path(args.participation_summary),
        representative_path=Path(args.representative_csv),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(out_path, index=False)
    feature_table.to_csv(out_path.with_suffix(".csv"), index=False)
    log(f"Saved DAO feature table: {out_path}")
    log(f"Saved DAO feature table csv: {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

