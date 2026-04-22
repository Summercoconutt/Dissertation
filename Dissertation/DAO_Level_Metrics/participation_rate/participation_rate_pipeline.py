from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datetime import datetime


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def parse_space_from_dirname(dirname: str) -> Optional[str]:
    # Accept both "space=<name>" and "<name>".
    if dirname.startswith("space="):
        return dirname.split("space=", 1)[1].strip()
    return dirname.strip() if dirname else None


def parse_proposal_id_from_filename(filename: str) -> str:
    # expected: proposal=<id>.parquet OR proposal=<id>.csv
    stem = Path(filename).stem
    if stem.startswith("proposal="):
        return stem.split("proposal=", 1)[1]
    return stem


def iter_proposal_files(base_spaces_dir: Path, spaces: Optional[Sequence[str]] = None) -> Iterable[Tuple[str, Path]]:
    spaces_set = set(spaces) if spaces else None
    for space_dir in sorted(base_spaces_dir.iterdir()):
        if not space_dir.is_dir():
            continue
        space = parse_space_from_dirname(space_dir.name)
        if not space:
            continue
        if spaces_set is not None and space not in spaces_set:
            continue

        # accept parquet or csv proposal files
        for p in sorted(space_dir.glob("proposal=*.parquet")):
            yield space, p
        for p in sorted(space_dir.glob("proposal=*.csv")):
            yield space, p


def _normalize_voter_vp_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "voter" not in df.columns and "Voter" in df.columns:
        df = df.rename(columns={"Voter": "voter"})
    if "vp" not in df.columns:
        if "Voting Power" in df.columns:
            df = df.rename(columns={"Voting Power": "vp"})
        elif "voting_power" in df.columns:
            df = df.rename(columns={"voting_power": "vp"})
    return df


def _read_minimal_columns(file_path: Path) -> pd.DataFrame:
    """
    Read only needed columns whenever possible for memory efficiency.
    """
    if file_path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(file_path, columns=["voter", "vp"])
        except Exception:
            try:
                return pd.read_parquet(file_path, columns=["Voter", "Voting Power"])
            except Exception:
                # Fallback to full read if schema varies
                return pd.read_parquet(file_path)
    if file_path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(file_path, usecols=["voter", "vp"])
        except Exception:
            try:
                return pd.read_csv(file_path, usecols=["Voter", "Voting Power"])
            except Exception:
                return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def load_data(
    base_spaces_dir: Path,
    spaces: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stream over proposal files and build:
    1) proposal_stats: one row per (space, proposal_id)
    2) active_voters_per_dao: one row per space

    This is memory-efficient and suitable for scaling to ~440 DAOs.
    """
    proposal_rows = []
    active_voter_sets: Dict[str, set] = {}
    failed_rows = []
    n_files = 0
    n_failed = 0

    for space, file_path in iter_proposal_files(base_spaces_dir, spaces=spaces):
        n_files += 1
        try:
            df = _read_minimal_columns(file_path)
            df = _normalize_voter_vp_columns(df)

            if "voter" not in df.columns or "vp" not in df.columns:
                warn(f"Skipping {file_path}: missing voter/vp columns.")
                n_failed += 1
                failed_rows.append({"space": space, "file_path": str(file_path), "reason": "missing voter/vp columns"})
                continue

            # row-level cleaning per proposal file
            df = df.dropna(subset=["voter", "vp"]).copy()
            df["vp"] = pd.to_numeric(df["vp"], errors="coerce")
            df = df.dropna(subset=["vp"])
            if df.empty:
                continue
            df = df.drop_duplicates(subset=["voter", "vp"])

            proposal_id = parse_proposal_id_from_filename(file_path.name)
            uniq_voters = pd.Series(df["voter"]).dropna().astype(str).unique()
            n_voters = int(len(uniq_voters))
            sum_actual_vp = float(df["vp"].sum())

            proposal_rows.append(
                {
                    "space": space,
                    "proposal_id": proposal_id,
                    "n_voters": n_voters,
                    "sum_actual_vp": sum_actual_vp,
                }
            )

            if space not in active_voter_sets:
                active_voter_sets[space] = set()
            active_voter_sets[space].update(uniq_voters.tolist())

            if n_files % 500 == 0:
                log(f"Processed files: {n_files:,}")
        except Exception as exc:
            warn(f"Failed reading {file_path}: {exc}")
            n_failed += 1
            failed_rows.append({"space": space, "file_path": str(file_path), "reason": str(exc)})

    if n_files == 0:
        raise FileNotFoundError(f"No proposal files found under {base_spaces_dir}")
    if not proposal_rows:
        raise RuntimeError("No valid proposal data loaded after cleaning.")

    proposal_stats = pd.DataFrame(proposal_rows).drop_duplicates(subset=["space", "proposal_id"])
    active_voters = pd.DataFrame(
        [{"space": s, "active_voters": len(voters)} for s, voters in active_voter_sets.items()]
    )
    log(
        f"Processed files: {n_files:,}, failed/skipped files: {n_failed:,}, "
        f"valid proposals: {len(proposal_stats):,}"
    )
    failed_df = pd.DataFrame(failed_rows)
    return proposal_stats, active_voters, failed_df


def compute_proposal_stats(proposal_stats: pd.DataFrame) -> pd.DataFrame:
    # Already proposal-level in streaming mode; keep function for pipeline clarity.
    return proposal_stats.copy()


def compute_dao_stats(active_voters: pd.DataFrame, proposal_stats: pd.DataFrame) -> pd.DataFrame:
    # max_vp and robust_max_vp computed from proposal sums within DAO
    g = proposal_stats.groupby("space")["sum_actual_vp"]
    dao_vp = g.agg(max_vp="max").reset_index()
    robust = g.quantile(0.95).reset_index(name="robust_max_vp")

    out = active_voters.merge(dao_vp, on="space", how="left").merge(robust, on="space", how="left")
    return out


def merge_stats(proposal_stats: pd.DataFrame, dao_stats: pd.DataFrame) -> pd.DataFrame:
    return proposal_stats.merge(dao_stats, on="space", how="left")


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace({0: np.nan})
    return numer / denom


def compute_participation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["participation_count"] = _safe_div(out["n_voters"], out["active_voters"])
    out["participation_vp"] = _safe_div(out["sum_actual_vp"], out["max_vp"])
    out["robust_participation_vp"] = _safe_div(out["sum_actual_vp"], out["robust_max_vp"])

    # edge cases: explicit None-like (NaN is fine for CSV)
    # clip to [0, 1.5]
    for col in ["participation_count", "participation_vp", "robust_participation_vp"]:
        out[col] = out[col].clip(lower=0.0, upper=1.5)

    out["vp_gt_1_flag"] = out["participation_vp"] > 1.0
    out["robust_vp_gt_1_flag"] = out["robust_participation_vp"] > 1.0
    return out


def aggregate_dao_level(proposal_level: pd.DataFrame) -> pd.DataFrame:
    g = proposal_level.groupby("space", dropna=False)
    out = g.agg(
        mean_participation_count=("participation_count", "mean"),
        std_participation_count=("participation_count", "std"),
        mean_participation_vp=("participation_vp", "mean"),
        std_participation_vp=("participation_vp", "std"),
        mean_robust_participation_vp=("robust_participation_vp", "mean"),
        std_robust_participation_vp=("robust_participation_vp", "std"),
        mean_n_voters=("n_voters", "mean"),
        mean_sum_vp=("sum_actual_vp", "mean"),
        n_proposals=("proposal_id", "nunique"),
    ).reset_index()
    return out


def print_diagnostics(proposal_level: pd.DataFrame, dao_level: pd.DataFrame) -> None:
    print("\n=== Diagnostics ===")
    print(f"Total DAOs: {proposal_level['space'].nunique():,}")
    print(f"Total proposals: {proposal_level[['space','proposal_id']].drop_duplicates().shape[0]:,}")

    # distributions
    for col in ["participation_count", "participation_vp"]:
        s = proposal_level[col].dropna()
        if len(s) == 0:
            print(f"{col}: (no non-null values)")
            continue
        print(f"\n{col} distribution:")
        print(s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())

    # top/bottom 10 DAOs by mean participation_count
    ranked = dao_level.sort_values("mean_participation_count", ascending=False)
    print("\nTop 10 DAOs by mean_participation_count:")
    print(ranked.head(10)[["space", "mean_participation_count", "n_proposals"]].to_string(index=False))
    print("\nBottom 10 DAOs by mean_participation_count:")
    print(ranked.tail(10)[["space", "mean_participation_count", "n_proposals"]].to_string(index=False))
    print("===================\n")


def save_outputs(
    proposal_level: pd.DataFrame,
    dao_level: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "proposal_level_participation.csv"
    p2 = out_dir / "dao_level_participation_summary.csv"

    proposal_cols = [
        "space",
        "proposal_id",
        "n_voters",
        "sum_actual_vp",
        "active_voters",
        "max_vp",
        "robust_max_vp",
        "participation_count",
        "participation_vp",
        "robust_participation_vp",
        "vp_gt_1_flag",
        "robust_vp_gt_1_flag",
    ]
    try:
        proposal_level.loc[:, proposal_cols].to_csv(p1, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p1 = out_dir / f"proposal_level_participation_{ts}.csv"
        warn(f"Target file locked. Writing proposal-level output to: {p1}")
        proposal_level.loc[:, proposal_cols].to_csv(p1, index=False)

    try:
        dao_level.to_csv(p2, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p2 = out_dir / f"dao_level_participation_summary_{ts}.csv"
        warn(f"Target file locked. Writing DAO-level output to: {p2}")
        dao_level.to_csv(p2, index=False)

    return p1, p2


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Snapshot DAO participation metrics.")
    parser.add_argument(
        "--base-spaces-dir",
        default=r"C:\Users\DELL\Desktop\snapshot_votes_data\snapshot_votes_441\spaces",
        help="Base directory containing space=... folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\DELL\Desktop\snapshot_votes_data\snapshot_votes_441\outputs\participation_rate",
        help="Output directory for CSV results.",
    )
    parser.add_argument(
        "--spaces",
        default="",
        help="Comma-separated space list for a sample run (e.g. uniswapgovernance.eth,aave.eth,ens.eth). Empty = all.",
    )
    parser.add_argument(
        "--timestamped-run-dir",
        action="store_true",
        help="If set, save outputs to out-dir/run_YYYYMMDD_HHMMSS instead of directly under out-dir.",
    )
    args = parser.parse_args()

    base_spaces_dir = Path(args.base_spaces_dir)
    out_dir = Path(args.out_dir)
    if args.timestamped_run_dir:
        run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir = out_dir / run_tag
    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()] or None

    log(f"Base spaces dir: {base_spaces_dir}")
    log(f"Output dir: {out_dir}")
    if spaces:
        log(f"Running sample spaces: {spaces}")

    proposal_stats_raw, active_voters, failed_files = load_data(base_spaces_dir, spaces=spaces)
    proposal_stats = compute_proposal_stats(proposal_stats_raw)
    dao_stats = compute_dao_stats(active_voters, proposal_stats)
    merged = merge_stats(proposal_stats, dao_stats)
    proposal_level = compute_participation_metrics(merged)
    dao_level = aggregate_dao_level(proposal_level)

    p1, p2 = save_outputs(proposal_level, dao_level, out_dir)
    if not failed_files.empty:
        failed_path = out_dir / "failed_files_log.csv"
        failed_files.to_csv(failed_path, index=False)
        log(f"Saved failed files log: {failed_path}")
    log(f"Saved proposal-level participation: {p1}")
    log(f"Saved DAO-level participation summary: {p2}")

    print_diagnostics(proposal_level, dao_level)


if __name__ == "__main__":
    main()

