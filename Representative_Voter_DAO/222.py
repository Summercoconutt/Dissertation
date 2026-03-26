from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

# ========================
# LOGGER
# ========================
def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

# ========================
# HELPERS
# ========================
def parse_space_from_folder_name(folder_name: str) -> Optional[str]:
    """
    Accepts both naming styles:
    - space=aave.eth
    - aave.eth
    """
    if folder_name.startswith("space="):
        return folder_name.split("space=", 1)[1].strip()
    return folder_name.strip()

def parse_proposal_id_from_file_name(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("proposal=", 1)[1] if stem.startswith("proposal=") else stem

# ========================
# LOAD DAO DATA
# ========================
def load_selected_spaces(base_spaces_path: Path, selected_spaces: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Recursively reads all DAO parquet files and merges them.
    If selected_spaces=None → load all automatically detected DAO folders.
    """
    frames: List[pd.DataFrame] = []
    selected_set = set(selected_spaces) if selected_spaces else None

    if not base_spaces_path.exists():
        raise FileNotFoundError(f"Base spaces path not found: {base_spaces_path}")

    detected_folders = [p for p in base_spaces_path.iterdir() if p.is_dir()]
    log(f"Detected folders: {[p.name for p in detected_folders]}")

    for space_dir in sorted(detected_folders):
        space_name = parse_space_from_folder_name(space_dir.name)
        if selected_set is not None and space_name not in selected_set:
            continue

        proposal_files = sorted(space_dir.glob("proposal=*.parquet"))
        if not proposal_files:
            warn(f"No proposal files found in {space_dir}")
            continue

        log(f"Reading {len(proposal_files)} proposals from {space_name}")
        for pf in proposal_files:
            try:
                df = pd.read_parquet(pf)
                df["space"] = space_name
                df["proposal_id"] = parse_proposal_id_from_file_name(pf.name)

                # Fix mixed-type errors in Choice
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
        raise RuntimeError("No parquet data loaded. Check directory structure.")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    log(f"Loaded DAOs: {sorted(merged['space'].unique().tolist())}")
    log(f"Total vote rows: {len(merged):,}")
    return merged

# ========================
# NORMALIZATION
# ========================
def normalize_choice(row: pd.Series) -> str:
    """
    Normalize vote choice into {"for","against","abstain","unknown"}.
    Handles number, string, dict representation.
    """
    choice_raw = row.get("Choice", np.nan)
    vote_label_raw = row.get("Vote Label", np.nan)

    try:
        if isinstance(choice_raw, str) and choice_raw.strip().startswith("{"):
            choice_raw = np.nan
        elif isinstance(choice_raw, str):
            choice_raw = float(choice_raw)
    except Exception:
        pass

    cnum = pd.to_numeric(pd.Series([choice_raw]), errors="coerce").iloc[0]
    if pd.notna(cnum):
        if int(cnum) == 1:
            return "for"
        if int(cnum) == 2:
            return "against"
        if int(cnum) == 3:
            return "abstain"

    if pd.notna(vote_label_raw):
        txt = str(vote_label_raw).lower().strip().replace("_", " ").replace("-", " ")
        if "against" in txt:
            return "against"
        if "abstain" in txt:
            return "abstain"
        if any(k in txt for k in ["for", "yes", "yay", "approve", "support", "approved"]):
            return "for"
    return "unknown"

# ========================
# METRIC COMPUTATION
# ========================
def compute_majority_choice(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["choice_norm"].isin(["for", "against", "abstain"])].copy()
    if valid.empty:
        df["majority_choice"] = "unknown"
        df["aligned_with_majority"] = False
        return df

    counts = valid.groupby(["proposal_id", "choice_norm"]).size().reset_index(name="n")
    rows = []
    for pid, g in counts.groupby("proposal_id"):
        g = g.sort_values("n", ascending=False)
        if len(g[g["n"] == g["n"].iloc[0]]) > 1:
            rows.append((pid, "unknown"))
        else:
            rows.append((pid, g.iloc[0]["choice_norm"]))
    majority_df = pd.DataFrame(rows, columns=["proposal_id", "majority_choice"])
    out = df.merge(majority_df, on="proposal_id", how="left")
    out["majority_choice"].fillna("unknown", inplace=True)
    out["aligned_with_majority"] = out["choice_norm"] == out["majority_choice"]
    return out

def compute_whale_flag(df: pd.DataFrame) -> pd.DataFrame:
    if "Voting Power" not in df.columns:
        df["is_whale"] = False
        return df
    vp = pd.to_numeric(df["Voting Power"], errors="coerce")
    q99 = vp.quantile(0.99)
    df["is_whale"] = vp >= q99 if pd.notna(q99) else False
    return df

def compute_avg_days_between_votes(vote_ts: pd.Series) -> float:
    ts = pd.to_datetime(vote_ts, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return np.nan
    return float(ts.diff().dropna().dt.total_seconds().mean() / 86400)

def compute_choice_entropy(choices: pd.Series) -> float:
    vals = choices[choices.isin(["for", "against", "abstain"])]
    if len(vals) == 0:
        return 0.0
    p = vals.value_counts(normalize=True).values
    return float(-np.sum(p * np.log(p)))

# ========================
# VOTER FEATURE TABLE
# ========================
def build_voter_base(df_votes: pd.DataFrame) -> pd.DataFrame:
    work = df_votes.copy().drop_duplicates().dropna(subset=["Voter"])

    if "Voting Power" not in work.columns:
        work["Voting Power"] = np.nan
    work["Voting Power"] = pd.to_numeric(work["Voting Power"], errors="coerce")
    work = work.dropna(subset=["Voting Power"])

    work["Vote Timestamp"] = pd.to_datetime(work.get("Vote Timestamp", pd.NaT), errors="coerce")

    records = []
    for (space, voter), g in work.groupby(["space", "Voter"]):
        total_votes = len(g)
        cnts = g["choice_norm"].value_counts()
        ts = g["Vote Timestamp"].dropna().sort_values()
        first, last = (ts.min(), ts.max()) if not ts.empty else (pd.NaT, pd.NaT)
        active_days = (last - first).total_seconds()/86400 if pd.notna(first) and pd.notna(last) else np.nan

        rec = {
            "space": space,
            "voter": voter,
            "total_votes": total_votes,
            "avg_voting_power": g["Voting Power"].mean(),
            "median_voting_power": g["Voting Power"].median(),
            "max_voting_power": g["Voting Power"].max(),
            "voting_power_std": g["Voting Power"].std(ddof=0),
            "for_votes": int(cnts.get("for", 0)),
            "against_votes": int(cnts.get("against", 0)),
            "abstain_votes": int(cnts.get("abstain", 0)),
            "pct_for_votes": cnts.get("for", 0) / total_votes,
            "pct_against_votes": cnts.get("against", 0) / total_votes,
            "pct_abstain_votes": cnts.get("abstain", 0) / total_votes,
            "pct_aligned_with_majority": g.get("aligned_with_majority", pd.Series(dtype=float)).astype(float).mean(),
            "is_whale_ratio": g.get("is_whale", pd.Series(dtype=float)).astype(float).mean(),
            "first_vote_timestamp": first,
            "last_vote_timestamp": last,
            "active_duration_days": active_days,
            "avg_days_between_votes": compute_avg_days_between_votes(g["Vote Timestamp"]),
            "choice_entropy": compute_choice_entropy(g["choice_norm"]),
        }
        records.append(rec)
    return pd.DataFrame.from_records(records)

# ========================
# NORMALIZATION FUNCTIONS
# ========================
def zscore_standardize(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = df.copy()
    med = x[cols].median(numeric_only=True)
    x[cols] = x[cols].fillna(med)
    mean = x[cols].mean(numeric_only=True)
    std = x[cols].std(ddof=0, numeric_only=True).replace(0, 1)
    x[cols] = (x[cols] - mean) / std
    return x, mean, std

# ========================
# DAO REPRESENTATIVE VOTERS
# ========================
def build_representative_voters(voter_base: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "total_votes", "avg_voting_power", "median_voting_power", "max_voting_power", "voting_power_std",
        "pct_for_votes", "pct_against_votes", "pct_abstain_votes", "pct_aligned_with_majority",
        "is_whale_ratio", "active_duration_days", "avg_days_between_votes", "choice_entropy",
    ]
    # Raw mean per DAO
    raw_centroid = (
        voter_base.groupby("space", dropna=False)[feature_cols]
        .mean()
        .add_prefix("raw_")
        .reset_index()
    )
    # Z-score mean per DAO
    scaled, mean, std = zscore_standardize(voter_base, feature_cols)
    z_centroid = (
        scaled.groupby("space", dropna=False)[feature_cols]
        .mean()
        .add_prefix("z_")
        .reset_index()
    )
    n_voters = voter_base.groupby("space")["voter"].nunique().reset_index(name="n_voters_in_dao")
    return raw_centroid.merge(z_centroid, on="space").merge(n_voters, on="space")

# ========================
# REPORT
# ========================
def print_summary(df_votes: pd.DataFrame, voter_base: pd.DataFrame) -> None:
    print("\n=== SUMMARY ===")
    print(f"DAOs loaded: {df_votes['space'].nunique()}")
    print(f"Proposals: {df_votes['proposal_id'].nunique()}")
    print(f"Total votes: {len(df_votes):,}")
    print(f"(space,voter) pairs: {voter_base[['space','voter']].drop_duplicates().shape[0]:,}")
    print("================\n")

# ========================
# MAIN PIPELINE
# ========================
def main() -> None:
    base_root = Path("D:/snapshot_votes_441")
    base_spaces_path = base_root / "spaces"
    out_dir = base_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Reading DAO files from {base_spaces_path}")
    merged = load_selected_spaces(base_spaces_path, selected_spaces=None)

    merged["Vote Timestamp"] = pd.to_datetime(merged.get("Vote Timestamp", pd.NaT), errors="coerce")
    merged["Created Time"] = pd.to_datetime(merged.get("Created Time", pd.NaT), errors="coerce")

    log("Normalizing choices...")
    merged["choice_norm"] = merged.apply(normalize_choice, axis=1)

    log("Computing majority choice...")
    merged = compute_majority_choice(merged)

    log("Computing whale flag...")
    merged = compute_whale_flag(merged)

    if "Choice" in merged.columns:
        merged["Choice"] = merged["Choice"].astype(str)

    merged_out = out_dir / "merged_votes_all_daos.parquet"
    merged.to_parquet(merged_out, index=False)
    log(f"Saved merged votes: {merged_out}")

    log("Building voter base...")
    voter_base = build_voter_base(merged)
    voter_base.to_parquet(out_dir / "voter_base_all_daos.parquet", index=False)
    voter_base.to_csv(out_dir / "voter_base_all_daos.csv", index=False)
    log("Saved voter base")

    log("Building representative DAO profiles...")
    dao_rep = build_representative_voters(voter_base)
    dao_rep.to_parquet(out_dir / "dao_representative_voters_all_daos.parquet", index=False)
    dao_rep.to_csv(out_dir / "dao_representative_voters_all_daos.csv", index=False)
    log("Saved DAO representative centroids (raw + z-score)")

    print_summary(merged, voter_base)
    log("🎯 Pipeline completed successfully.")

# ========================
if __name__ == "__main__":
    main()
