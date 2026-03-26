from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Easy-to-edit list of selected DAOs (10 by default)
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
    if not folder_name.startswith("space="):
        return None
    return folder_name.split("space=", 1)[1].strip()


def parse_proposal_id_from_file_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem.startswith("proposal="):
        return stem.split("proposal=", 1)[1]
    return stem


def load_selected_spaces(base_spaces_path: Path, selected_spaces: Sequence[str]) -> pd.DataFrame:
    """
    Recursively read proposal parquet files from selected DAO folders and merge.
    Adds `space` and `proposal_id` columns from path metadata.
    """
    frames: List[pd.DataFrame] = []
    selected_set = set(selected_spaces)

    if not base_spaces_path.exists():
        raise FileNotFoundError(f"Base spaces path not found: {base_spaces_path}")

    for space_dir in sorted(base_spaces_path.iterdir()):
        if not space_dir.is_dir():
            continue

        space_name = parse_space_from_folder_name(space_dir.name)
        if space_name is None or space_name not in selected_set:
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

                # --- ✅ Fix mixed-type column "Choice" here ---
                if "Choice" in df.columns:
                    df["Choice"] = df["Choice"].apply(
                        lambda x: str(x)
                        if isinstance(x, (dict, list))
                        else (np.nan if x in ["", None] else x)
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
    Supports multiple data types: int, str, dict/json, etc.
    """
    choice_raw = row.get("Choice", np.nan)
    vote_label_raw = row.get("Vote Label", np.nan)

    # Handle dict-like strings (e.g. '{"1":55,"2":4}')
    if isinstance(choice_raw, str):
        if choice_raw.strip().startswith("{") and choice_raw.strip().endswith("}"):
            choice_raw = np.nan
        else:
            try:
                choice_raw = float(choice_raw)
            except Exception:
                pass

    # Numeric mapping
    cnum = pd.to_numeric(pd.Series([choice_raw]), errors="coerce").iloc[0]
    if pd.notna(cnum):
        if int(cnum) == 1:
            return "for"
        if int(cnum) == 2:
            return "against"
        if int(cnum) == 3:
            return "abstain"

    # Fallback textual label
    if pd.notna(vote_label_raw):
        txt = str(vote_label_raw).lower().strip()
        txt = txt.replace("_", " ").replace("-", " ")
        if "against" in txt:
            return "against"
        if "abstain" in txt:
            return "abstain"
        if any(k in txt for k in ["for", "yes", "yay", "approve", "support", "approved"]):
            return "for"

    return "unknown"


def compute_majority_choice(df: pd.DataFrame) -> pd.DataFrame:
    if "proposal_id" not in df.columns or "choice_norm" not in df.columns:
        raise ValueError("Missing essential columns for majority computation")

    valid = df[df["choice_norm"].isin(["for", "against", "abstain"])].copy()
    if valid.empty:
        warn("No valid votes. Setting majority_choice=unknown")
        df["majority_choice"] = "unknown"
        df["aligned_with_majority"] = False
        return df

    counts = valid.groupby(["proposal_id", "choice_norm"], dropna=False).size().reset_index(name="n")

    majority_rows = []
    for pid, g in counts.groupby("proposal_id"):
        g = g.sort_values("n", ascending=False)
        if len(g[g["n"] == g["n"].iloc[0]]) > 1:
            majority_rows.append((pid, "unknown"))
        else:
            majority_rows.append((pid, g["choice_norm"].iloc[0]))

    majority_df = pd.DataFrame(majority_rows, columns=["proposal_id", "majority_choice"])
    out = df.merge(majority_df, how="left", on="proposal_id")
    out["majority_choice"] = out["majority_choice"].fillna("unknown")
    out["aligned_with_majority"] = out["choice_norm"] == out["majority_choice"]
    return out


def compute_whale_flag(df: pd.DataFrame) -> pd.DataFrame:
    if "Voting Power" not in df.columns:
        warn("Missing Voting Power; defaulting all to False")
        df["is_whale"] = False
        return df

    vp = pd.to_numeric(df["Voting Power"], errors="coerce")
    q99 = vp.quantile(0.99)
    if pd.isna(q99):
        warn("Cannot compute whale threshold; default False")
        df["is_whale"] = False
        return df

    df["is_whale"] = vp >= q99
    return df


def compute_avg_days_between_votes(vote_ts: pd.Series) -> float:
    ts = pd.to_datetime(vote_ts, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return np.nan
    diffs = ts.diff().dropna().dt.total_seconds() / 86400.0
    return float(diffs.mean()) if len(diffs) > 0 else np.nan


def compute_choice_entropy(choices: pd.Series) -> float:
    vals = choices[choices.isin(["for", "against", "abstain"])]
    if len(vals) == 0:
        return 0.0
    p = vals.value_counts(normalize=True).values
    return float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))


def build_voter_base(df_votes: pd.DataFrame) -> pd.DataFrame:
    required = {"space", "Voter", "proposal_id"}
    missing = required - set(df_votes.columns)
    if missing:
        raise ValueError(f"Missing {missing}")

    work = df_votes.copy()
    work = work.drop_duplicates()
    work = work.dropna(subset=["Voter"])

    if "Voting Power" not in work.columns:
        warn("Adding NaN Voting Power column")
        work["Voting Power"] = np.nan
    work["Voting Power"] = pd.to_numeric(work["Voting Power"], errors="coerce")
    work = work.dropna(subset=["Voting Power"])

    if "Vote Timestamp" in work.columns:
        work["Vote Timestamp"] = pd.to_datetime(work["Vote Timestamp"], errors="coerce")
    else:
        work["Vote Timestamp"] = pd.NaT

    records = []
    for (space, voter), g in work.groupby(["space", "Voter"], dropna=False):
        total_votes = len(g)
        choice_counts = g["choice_norm"].value_counts()
        ts_sorted = g["Vote Timestamp"].dropna().sort_values()

        first_ts = ts_sorted.min() if not ts_sorted.empty else pd.NaT
        last_ts = ts_sorted.max() if not ts_sorted.empty else pd.NaT
        active_days = (
            (last_ts - first_ts).total_seconds() / 86400 if pd.notna(first_ts) and pd.notna(last_ts) else np.nan
        )

        rec = {
            "space": space,
            "voter": voter,
            "total_votes": total_votes,
            "avg_voting_power": g["Voting Power"].mean(),
            "median_voting_power": g["Voting Power"].median(),
            "max_voting_power": g["Voting Power"].max(),
            "voting_power_std": g["Voting Power"].std(ddof=0),
            "for_votes": int(choice_counts.get("for", 0)),
            "against_votes": int(choice_counts.get("against", 0)),
            "abstain_votes": int(choice_counts.get("abstain", 0)),
            "pct_for_votes": choice_counts.get("for", 0) / total_votes,
            "pct_against_votes": choice_counts.get("against", 0) / total_votes,
            "pct_abstain_votes": choice_counts.get("abstain", 0) / total_votes,
            "pct_aligned_with_majority": g.get("aligned_with_majority", pd.Series(dtype=float)).astype(float).mean()
            if "aligned_with_majority" in g.columns
            else np.nan,
            "is_whale_ratio": g.get("is_whale", pd.Series(dtype=float)).astype(float).mean()
            if "is_whale" in g.columns
            else np.nan,
            "first_vote_timestamp": first_ts,
            "last_vote_timestamp": last_ts,
            "active_duration_days": active_days,
            "avg_days_between_votes": compute_avg_days_between_votes(g["Vote Timestamp"]),
            "choice_entropy": compute_choice_entropy(g["choice_norm"]),
        }
        records.append(rec)

    return pd.DataFrame.from_records(records)


def zscore_standardize(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = df.copy()
    med = x[list(cols)].median(numeric_only=True)
    x[list(cols)] = x[list(cols)].fillna(med)
    mean = x[list(cols)].mean(numeric_only=True)
    std = x[list(cols)].std(ddof=0, numeric_only=True).replace(0, 1)
    x[list(cols)] = (x[list(cols)] - mean) / std
    return x, mean, std


def build_representative_voters(voter_base: pd.DataFrame) -> pd.DataFrame:
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
    scaled, _, _ = zscore_standardize(voter_base, feature_cols)
    centroid = (
        scaled.groupby("space", dropna=False)[feature_cols].mean().add_prefix("rep_").reset_index()
    )
    n_voters = voter_base.groupby("space", dropna=False)["voter"].nunique().reset_index(name="n_voters_in_dao")
    return centroid.merge(n_voters, on="space", how="left")


def print_summary_stats(df_votes: pd.DataFrame, voter_base: pd.DataFrame) -> None:
    n_daos = df_votes["space"].nunique(dropna=True)
    n_proposals = df_votes["proposal_id"].nunique(dropna=True)
    n_votes = len(df_votes)
    n_pairs = voter_base[["space", "voter"]].drop_duplicates().shape[0]
    print("\n=== Summary ===")
    print(f"DAOs: {n_daos}")
    print(f"Proposals: {n_proposals}")
    print(f"Votes: {n_votes}")
    print(f"(space,voter) pairs: {n_pairs}")
    print("==============\n")


def main() -> None:
    base_root = Path("D:/snapshot_votes_441")
    base_spaces_path = base_root / "spaces"
    out_dir = base_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Reading from {base_spaces_path}")
    merged = load_selected_spaces(base_spaces_path, SELECTED_SPACES)

    # Ensures timestamp parsing
    merged["Vote Timestamp"] = pd.to_datetime(merged.get("Vote Timestamp", pd.NaT), errors="coerce")
    merged["Created Time"] = pd.to_datetime(merged.get("Created Time", pd.NaT), errors="coerce")

    log("Normalizing choice labels...")
    merged["choice_norm"] = merged.apply(normalize_choice, axis=1)

    log("Computing majority choice...")
    merged = compute_majority_choice(merged)

    log("Computing whale flag (top 1%)...")
    merged = compute_whale_flag(merged)

    # ✅ Fix parquet saving type issue
    if "Choice" in merged.columns:
        merged["Choice"] = merged["Choice"].astype(str)

    merged_out = out_dir / "merged_votes_10daos.parquet"
    merged.to_parquet(merged_out, index=False)
    log(f"Saved merged votes to {merged_out}")

    log("Building voter base...")
    voter_base = build_voter_base(merged)
    voter_base.to_parquet(out_dir / "voter_base_10daos.parquet", index=False)
    voter_base.to_csv(out_dir / "voter_base_10daos.csv", index=False)
    log("Saved voter base")

    log("Computing representative DAO centroids...")
    dao_rep = build_representative_voters(voter_base)
    dao_rep.to_parquet(out_dir / "dao_representative_voters_10daos.parquet", index=False)
    dao_rep.to_csv(out_dir / "dao_representative_voters_10daos.csv", index=False)
    log("Saved DAO-level centroids")

    print_summary_stats(merged, voter_base)
    log("🎯 Pipeline completed successfully.")


if __name__ == "__main__":
    main()
