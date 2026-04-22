from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Easy-to-edit default list (used when --all-spaces is not set and --spaces not provided)
SELECTED_SPACES: List[str] = [
    "uniswapgovernance.eth",
    "aave.eth",
    "ens.eth",
    "balancer.eth",
    "curve.eth",
    "lido-snapshot.eth",
    "sushigov.eth",
    "opcollective.eth",
    "arbitrumfoundation.eth",
    "gitcoindao.eth",
]


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def parse_space_from_folder_name(folder_name: str) -> Optional[str]:
    # Accept both "space=<space_name>" and "<space_name>"
    if folder_name.startswith("space="):
        return folder_name.split("space=", 1)[1].strip()
    return folder_name.strip()


def parse_proposal_id_from_file_name(file_name: str) -> str:
    # Expected pattern: proposal=<id>.parquet
    stem = Path(file_name).stem
    if stem.startswith("proposal="):
        return stem.split("proposal=", 1)[1]
    return stem


def load_selected_spaces(base_spaces_path: Path, selected_spaces: Optional[Sequence[str]]) -> pd.DataFrame:
    """
    Recursively read proposal parquet files from selected DAO folders and merge.
    Adds `space` and `proposal_id` columns from path metadata.
    """
    frames: List[pd.DataFrame] = []
    selected_set = set(selected_spaces) if selected_spaces is not None else None

    if not base_spaces_path.exists():
        raise FileNotFoundError(f"Base spaces path not found: {base_spaces_path}")

    for space_dir in sorted(base_spaces_path.iterdir()):
        if not space_dir.is_dir():
            continue

        space_name = parse_space_from_folder_name(space_dir.name)
        if space_name is None:
            continue
        if selected_set is not None and space_name not in selected_set:
            continue

        proposal_files = sorted(space_dir.glob("proposal=*.parquet"))
        if not proposal_files:
            warn(f"No proposal parquet files found under {space_dir}")
            continue

        log(f"Reading {len(proposal_files)} proposal files from {space_name}")
        for pf in proposal_files:
            try:
                df = pd.read_parquet(pf)
                df["space"] = space_name
                df["proposal_id"] = parse_proposal_id_from_file_name(pf.name)

                # Normalize mixed-type Choice values early to prevent Arrow conversion issues.
                if "Choice" in df.columns:
                    df["Choice"] = df["Choice"].apply(
                        lambda x: str(x) if isinstance(x, (dict, list)) else (np.nan if x in ["", None] else x)
                    )
                frames.append(df)
            except Exception as exc:
                warn(f"Failed to read {pf}: {exc}")

    if not frames:
        raise RuntimeError("No parquet data loaded. Check selected spaces and file paths.")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    log(f"Merged vote rows: {len(merged):,}")
    return merged


def normalize_choice(row: pd.Series) -> str:
    """
    Normalize vote choice into {"for","against","abstain","unknown"}.
    Priority:
    1) numeric Choice mapping 1/2/3
    2) text in Vote Label fallback
    """
    choice_raw = row.get("Choice", np.nan)
    vote_label_raw = row.get("Vote Label", np.nan)

    # First try numeric Choice
    cnum = pd.to_numeric(pd.Series([choice_raw]), errors="coerce").iloc[0]
    if pd.notna(cnum):
        cnum_int = int(cnum)
        if cnum_int == 1:
            return "for"
        if cnum_int == 2:
            return "against"
        if cnum_int == 3:
            return "abstain"

    # Fallback: Vote Label text
    if pd.notna(vote_label_raw):
        txt = str(vote_label_raw).strip().lower()
        txt = txt.replace("_", " ").replace("-", " ")
        if "against" in txt:
            return "against"
        if "abstain" in txt:
            return "abstain"
        if txt in {"for", "yes", "yay", "approve", "approved", "support"}:
            return "for"
        if "for" in txt:
            return "for"

    return "unknown"


def compute_majority_choice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute majority choice and aligned_with_majority by proposal_id.
    For ties or all unknown, set majority as unknown.
    """
    if "proposal_id" not in df.columns:
        raise ValueError("Essential column missing: proposal_id")
    if "choice_norm" not in df.columns:
        raise ValueError("Essential column missing: choice_norm")

    valid = df[df["choice_norm"].isin(["for", "against", "abstain"])].copy()
    if valid.empty:
        warn("No valid choice_norm values found; majority_choice set to unknown.")
        df["majority_choice"] = "unknown"
        df["aligned_with_majority"] = False
        return df

    counts = (
        valid.groupby(["proposal_id", "choice_norm"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    # Per proposal decide majority with tie handling
    majority_rows = []
    for pid, g in counts.groupby("proposal_id"):
        g = g.sort_values("n", ascending=False)
        top_n = g["n"].iloc[0]
        top = g[g["n"] == top_n]
        if len(top) > 1:
            majority_rows.append((pid, "unknown"))
        else:
            majority_rows.append((pid, top["choice_norm"].iloc[0]))
    majority_df = pd.DataFrame(majority_rows, columns=["proposal_id", "majority_choice"])

    out = df.merge(majority_df, how="left", on="proposal_id")
    out["majority_choice"] = out["majority_choice"].fillna("unknown")
    out["aligned_with_majority"] = out["choice_norm"] == out["majority_choice"]
    return out


def compute_whale_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute or overwrite is_whale using top 1% Voting Power threshold globally.
    """
    if "Voting Power" not in df.columns:
        warn("Missing 'Voting Power'. Cannot compute whale flag; set to False.")
        df["is_whale"] = False
        return df

    vp = pd.to_numeric(df["Voting Power"], errors="coerce")
    q99 = vp.quantile(0.99)
    if pd.isna(q99):
        warn("Voting Power quantile is NaN; whale flag set to False.")
        df["is_whale"] = False
        return df

    df["is_whale"] = vp >= q99
    return df


def compute_avg_days_between_votes(vote_ts: pd.Series) -> float:
    ts = pd.to_datetime(vote_ts, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return np.nan
    diffs = ts.diff().dropna().dt.total_seconds() / 86400.0
    if len(diffs) == 0:
        return np.nan
    return float(diffs.mean())


def compute_choice_entropy(choices: pd.Series) -> float:
    """
    Entropy over {for, against, abstain} distribution.
    """
    vals = choices[choices.isin(["for", "against", "abstain"])]
    if len(vals) == 0:
        return 0.0
    p = vals.value_counts(normalize=True).values.astype(float)
    # Safe entropy: -sum(p log p), ignoring zero probs
    entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
    return float(entropy)


def build_voter_base(df_votes: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vote-level data to (space, voter) feature table.
    """
    required = {"space", "Voter", "proposal_id"}
    missing = required - set(df_votes.columns)
    if missing:
        raise ValueError(f"Essential columns missing for voter base: {missing}")

    work = df_votes.copy()

    # Quality checks
    before = len(work)
    work = work.drop_duplicates()
    log(f"Removed exact duplicate rows: {before - len(work):,}")

    before = len(work)
    work = work.dropna(subset=["Voter"])
    log(f"Dropped rows with missing Voter: {before - len(work):,}")

    if "Voting Power" not in work.columns:
        warn("Missing Voting Power, creating as NaN.")
        work["Voting Power"] = np.nan
    work["Voting Power"] = pd.to_numeric(work["Voting Power"], errors="coerce")
    before = len(work)
    work = work.dropna(subset=["Voting Power"])
    log(f"Dropped rows with invalid Voting Power: {before - len(work):,}")

    # Ensure timestamps are datetime
    if "Vote Timestamp" in work.columns:
        work["Vote Timestamp"] = pd.to_datetime(work["Vote Timestamp"], errors="coerce")
    else:
        warn("Missing Vote Timestamp; time features will be mostly NaN.")
        work["Vote Timestamp"] = pd.NaT

    group_keys = ["space", "Voter"]
    records = []

    for (space, voter), g in work.groupby(group_keys, dropna=False):
        total_votes = len(g)
        choice_counts = g["choice_norm"].value_counts()

        for_votes = int(choice_counts.get("for", 0))
        against_votes = int(choice_counts.get("against", 0))
        abstain_votes = int(choice_counts.get("abstain", 0))

        ts_sorted = g["Vote Timestamp"].dropna().sort_values()
        first_ts = ts_sorted.iloc[0] if len(ts_sorted) > 0 else pd.NaT
        last_ts = ts_sorted.iloc[-1] if len(ts_sorted) > 0 else pd.NaT
        active_duration = (
            (last_ts - first_ts).total_seconds() / 86400.0 if pd.notna(first_ts) and pd.notna(last_ts) else np.nan
        )

        rec = {
            "space": space,
            "voter": voter,
            "total_votes": total_votes,
            "avg_voting_power": float(g["Voting Power"].mean()),
            "median_voting_power": float(g["Voting Power"].median()),
            "max_voting_power": float(g["Voting Power"].max()),
            "voting_power_std": float(g["Voting Power"].std(ddof=0)),
            "for_votes": for_votes,
            "against_votes": against_votes,
            "abstain_votes": abstain_votes,
            "pct_for_votes": for_votes / total_votes if total_votes else np.nan,
            "pct_against_votes": against_votes / total_votes if total_votes else np.nan,
            "pct_abstain_votes": abstain_votes / total_votes if total_votes else np.nan,
            "pct_aligned_with_majority": float(g["aligned_with_majority"].astype(float).mean())
            if "aligned_with_majority" in g.columns
            else np.nan,
            "is_whale_ratio": float(g["is_whale"].astype(float).mean()) if "is_whale" in g.columns else np.nan,
            "first_vote_timestamp": first_ts,
            "last_vote_timestamp": last_ts,
            "active_duration_days": active_duration,
            "avg_days_between_votes": compute_avg_days_between_votes(g["Vote Timestamp"]),
            "choice_entropy": compute_choice_entropy(g["choice_norm"]),
        }
        records.append(rec)

    voter_base = pd.DataFrame.from_records(records)
    return voter_base


def zscore_standardize(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Standardize columns with z-score. Missing values are median-imputed first.
    If std is 0, use std=1 to avoid division by zero.
    """
    x = df.copy()
    med = x[list(cols)].median(numeric_only=True)
    x[list(cols)] = x[list(cols)].fillna(med)
    mean = x[list(cols)].mean(numeric_only=True)
    std = x[list(cols)].std(ddof=0, numeric_only=True).replace(0, 1.0)
    x[list(cols)] = (x[list(cols)] - mean) / std
    return x, mean, std


def build_representative_voters(voter_base: pd.DataFrame) -> pd.DataFrame:
    """
    Build DAO-level representative voter centroids in two spaces:
    1) raw_rep_*: mean of raw voter features within each DAO
    2) z_rep_*: mean of z-scored voter features within each DAO
    """
    feature_cols = [
        "total_votes",
        "avg_voting_power",
        "median_voting_power",
        "max_voting_power",
        "voting_power_std",
        "pct_for_votes",
        "pct_against_votes",
        "pct_abstain_votes",
        "pct_aligned_with_majority",
        "is_whale_ratio",
        "active_duration_days",
        "avg_days_between_votes",
        "choice_entropy",
    ]

    missing = [c for c in feature_cols if c not in voter_base.columns]
    if missing:
        raise ValueError(f"Missing required voter feature columns: {missing}")

    raw_centroid = (
        voter_base.groupby("space", dropna=False)[feature_cols]
        .mean()
        .add_prefix("raw_rep_")
        .reset_index()
    )

    scaled, _, _ = zscore_standardize(voter_base, feature_cols)
    z_centroid = (
        scaled.groupby("space", dropna=False)[feature_cols]
        .mean()
        .add_prefix("z_rep_")
        .reset_index()
    )

    n_voters = voter_base.groupby("space", dropna=False)["voter"].nunique().reset_index(name="n_voters_in_dao")
    out = raw_centroid.merge(z_centroid, on="space", how="inner").merge(n_voters, on="space", how="left")
    return out


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing non-essential columns with safe defaults and warnings.
    """
    expected = [
        "Proposal Title",
        "Proposal Body",
        "Created Time",
        "Voter",
        "Original Choice",
        "Choice",
        "Vote Label",
        "Voting Power",
        "VP Ratio (%)",
        "Is Whale",
        "Aligned With Majority",
        "Vote Timestamp",
        "space",
        "proposal_id",
    ]
    for col in expected:
        if col not in df.columns:
            warn(f"Missing column '{col}', creating as NaN.")
            df[col] = np.nan
    return df


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert problematic mixed object columns to stable types before parquet write.
    This avoids pyarrow failures on columns containing dict/str/int mixtures.
    """
    out = df.copy()
    # Explicitly sanitize known problematic column first.
    if "Choice" in out.columns:
        out["Choice"] = out["Choice"].apply(lambda x: np.nan if pd.isna(x) else str(x))

    # Generic fallback for object columns that contain dict/list/tuple/set.
    for col in out.columns:
        if out[col].dtype == "object":
            sample = out[col].dropna().head(200)
            if sample.empty:
                continue
            has_nested = sample.map(lambda x: isinstance(x, (dict, list, tuple, set))).any()
            if has_nested:
                out[col] = out[col].apply(lambda x: np.nan if pd.isna(x) else str(x))
    return out


def print_summary_stats(df_votes: pd.DataFrame, voter_base: pd.DataFrame) -> None:
    n_daos = df_votes["space"].nunique(dropna=True) if "space" in df_votes.columns else 0
    n_proposals = df_votes["proposal_id"].nunique(dropna=True) if "proposal_id" in df_votes.columns else 0
    n_votes = len(df_votes)
    n_pairs = voter_base[["space", "voter"]].drop_duplicates().shape[0] if not voter_base.empty else 0

    print("\n=== Processing Summary ===")
    print(f"Number of DAOs: {n_daos:,}")
    print(f"Number of proposals: {n_proposals:,}")
    print(f"Number of vote rows: {n_votes:,}")
    print(f"Number of unique (space, voter) pairs: {n_pairs:,}")
    print("==========================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Representative voter pipeline for Snapshot DAOs.")
    parser.add_argument(
        "--base-root",
        default="D:/snapshot_votes_441",
        help="Base root folder containing spaces/ and outputs/.",
    )
    parser.add_argument(
        "--all-spaces",
        action="store_true",
        help="If set, process all detected spaces under spaces/.",
    )
    parser.add_argument(
        "--spaces",
        default="",
        help="Comma-separated explicit spaces list. Overrides default selected list.",
    )
    args = parser.parse_args()

    # Easy-to-modify paths
    base_root = Path(args.base_root)
    base_spaces_path = base_root / "spaces"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_root / "outputs" / f"DAO representative_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_spaces = [s.strip() for s in args.spaces.split(",") if s.strip()]
    if args.all_spaces:
        selected_spaces: Optional[Sequence[str]] = None
    elif custom_spaces:
        selected_spaces = custom_spaces
    else:
        selected_spaces = SELECTED_SPACES

    log(f"Base spaces path: {base_spaces_path}")
    log(f"Output path: {out_dir}")
    if selected_spaces is None:
        log("Selected spaces: ALL detected spaces")
    else:
        log(f"Selected spaces count: {len(selected_spaces)}")

    # 1) Load and merge selected DAO proposal files
    merged = load_selected_spaces(base_spaces_path, selected_spaces)
    merged = ensure_expected_columns(merged)

    # 2) Parse timestamps safely
    merged["Vote Timestamp"] = pd.to_datetime(merged["Vote Timestamp"], errors="coerce")
    merged["Created Time"] = pd.to_datetime(merged["Created Time"], errors="coerce")

    # 3) Normalize choices
    log("Normalizing vote choices...")
    merged["choice_norm"] = merged.apply(normalize_choice, axis=1)

    # 4) Majority choice and alignment (computed version preferred)
    log("Computing majority choice and aligned_with_majority...")
    merged = compute_majority_choice(merged)

    # 5) Whale indicator (computed version preferred)
    log("Computing is_whale flag from Voting Power 99th percentile...")
    merged = compute_whale_flag(merged)
    merged = make_arrow_safe(merged)

    # Save merged raw data
    merged_out = out_dir / "merged_votes_10daos.parquet"
    merged.to_parquet(merged_out, index=False)
    log(f"Saved merged raw data: {merged_out}")

    # 6) Build voter base at (space, voter)
    log("Building voter_base at (space, voter) level...")
    voter_base = build_voter_base(merged)

    # Save voter base
    vb_parquet = out_dir / "voter_base_10daos.parquet"
    vb_csv = out_dir / "voter_base_10daos.csv"
    voter_base.to_parquet(vb_parquet, index=False)
    voter_base.to_csv(vb_csv, index=False)
    log(f"Saved voter base parquet: {vb_parquet}")
    log(f"Saved voter base csv: {vb_csv}")

    # 7) Representative voter centroids per DAO
    log("Building representative voter (DAO centroids)...")
    dao_rep = build_representative_voters(voter_base)

    rep_parquet = out_dir / "dao_representative_voters_10daos.parquet"
    rep_csv = out_dir / "dao_representative_voters_10daos.csv"
    dao_rep.to_parquet(rep_parquet, index=False)
    dao_rep.to_csv(rep_csv, index=False)
    log(f"Saved representative DAO parquet: {rep_parquet}")
    log(f"Saved representative DAO csv: {rep_csv}")

    # Summary stats
    print_summary_stats(merged, voter_base)
    log("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

