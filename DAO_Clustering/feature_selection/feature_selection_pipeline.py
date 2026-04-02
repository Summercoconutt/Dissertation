"""
DAO clustering: full feature preprocessing + correlation-based feature selection.

Reads demand.txt specification (steps 0–11). Outputs under ./outputs/feature_screening/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# -----------------------------------------------------------------------------
# Feature groups (demand §2.1)
# -----------------------------------------------------------------------------

SCALE_FEATURES = [
    "n_unique_voters",
    "total_votes",
    "n_proposals",
    "mean_n_voters",
    "mean_sum_vp",
    "n_voters_in_dao",
]

PARTICIPATION_FEATURES = [
    "voter_turnout_proxy",
    "avg_voters_per_proposal",
    "median_voters_per_proposal",
    "mean_participation_count",
    "std_participation_count",
    "mean_participation_vp",
    "std_participation_vp",
    "mean_robust_participation_vp",
    "std_robust_participation_vp",
]

GOVERNANCE_FEATURES = [
    "gini_voting_power",
    "hhi_voting_power",
    "whale_ratio_top1pct",
]

ACTIVITY_FEATURES = [
    "proposal_frequency_per_30d",
    "repeat_voter_rate",
]

LOG_SCALE_TARGETS = [
    "n_unique_voters",
    "total_votes",
    "n_proposals",
    "mean_sum_vp",
    "mean_n_voters",
    "n_voters_in_dao",
]

# z_rep behavioural priorities (demand §9.5)
Z_REP_BEHAVIOUR_SUFFIXES = [
    "pct_for_votes",
    "pct_against_votes",
    "pct_abstain_votes",
    "choice_entropy",
    "pct_aligned_with_majority",
]


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx requires openpyxl. Install with: pip install openpyxl"
            ) from exc
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path}")


def separate_id_and_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "space" not in df.columns:
        raise ValueError("Column 'space' is required as identifier.")
    df_id = df[["space"]].copy()
    num = df.drop(columns=["space"]).select_dtypes(include=[np.number]).copy()
    return df_id, num


def report_missing(df: pd.DataFrame) -> pd.Series:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return miss


def basic_cleaning(df_num: pd.DataFrame, missing_threshold: float = 0.05) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Median impute if missing < threshold; drop column if >= threshold."""
    n = len(df_num)
    dropped_high_missing: List[str] = []
    out = df_num.copy()
    for col in out.columns:
        frac = out[col].isna().sum() / max(n, 1)
        if frac >= missing_threshold:
            out = out.drop(columns=[col])
            dropped_high_missing.append(col)
            warn(f"Dropped column (missing >= {missing_threshold:.0%}): {col} ({frac:.2%})")
        elif out[col].isna().any():
            med = out[col].median()
            out[col] = out[col].fillna(med)

    # constant columns
    const_cols = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
    if const_cols:
        out = out.drop(columns=const_cols)
        warn(f"Removed constant columns: {const_cols}")
    return out, dropped_high_missing, const_cols


def find_columns_by_prefix(cols: Sequence[str], prefix: str) -> List[str]:
    return [c for c in cols if c.startswith(prefix)]


def apply_group_preferences(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    §2.2: Prefer robust participation over non-robust; prefer z_rep over raw_rep.
    """
    removed: List[str] = []
    out = df.copy()
    cols = set(out.columns)

    # Drop non-robust VP if robust exists
    if "mean_robust_participation_vp" in cols:
        for c in ("mean_participation_vp", "std_participation_vp"):
            if c in cols:
                out = out.drop(columns=[c])
                removed.append(c)
    # Drop raw_rep_* when z_rep_* exists for same suffix
    raw_cols = find_columns_by_prefix(out.columns, "raw_rep_")
    for rc in raw_cols:
        suffix = rc.replace("raw_rep_", "", 1)
        zc = f"z_rep_{suffix}"
        if zc in out.columns:
            out = out.drop(columns=[rc])
            removed.append(rc)

    return out, removed


def add_log_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """§3: log1p on scale features; creates log_* columns."""
    out = df.copy()
    created: List[str] = []
    for base in LOG_SCALE_TARGETS:
        if base not in out.columns:
            continue
        new_name = f"log_{base}"
        s = pd.to_numeric(out[base], errors="coerce").clip(lower=0)
        out[new_name] = np.log1p(s)
        created.append(new_name)
    return out, created


def winsorize_df(df: pd.DataFrame, low: float = 0.01, high: float = 0.99) -> Tuple[pd.DataFrame, List[str]]:
    """§4: clip each column at low/high quantiles."""
    out = df.copy()
    affected: List[str] = []
    for c in out.columns:
        s = pd.to_numeric(out[c], errors="coerce")
        lo = s.quantile(low)
        hi = s.quantile(high)
        if pd.isna(lo) or pd.isna(hi):
            continue
        before = s.copy()
        clipped = s.clip(lower=lo, upper=hi)
        if not np.allclose(before.fillna(0), clipped.fillna(0), equal_nan=True):
            affected.append(c)
        out[c] = clipped
    return out, affected


def robust_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, RobustScaler]:
    """§5: RobustScaler on all numeric features."""
    scaler = RobustScaler()
    arr = scaler.fit_transform(df.to_numpy(dtype=float))
    scaled = pd.DataFrame(arr, columns=df.columns, index=df.index)
    return scaled, scaler


def correlation_pairs(corr: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """§7: pairs with |r| > threshold (upper triangle)."""
    cols = corr.columns.tolist()
    rows = []
    for i, ci in enumerate(cols):
        for j in range(i + 1, len(cols)):
            cj = cols[j]
            v = corr.loc[ci, cj]
            if pd.notna(v) and abs(v) > threshold:
                rows.append({"feature_a": ci, "feature_b": cj, "correlation": float(v)})
    return pd.DataFrame(rows)


def mark_redundant_from_pairs(pairs: pd.DataFrame, priority_order: Sequence[str]) -> List[str]:
    """Greedy: drop lower-priority feature in each highly correlated pair."""
    priority = {name: i for i, name in enumerate(priority_order)}
    to_drop: Set[str] = set()
    for _, row in pairs.iterrows():
        a, b = row["feature_a"], row["feature_b"]
        pa = priority.get(a, 10**6)
        pb = priority.get(b, 10**6)
        if pa < pb:
            to_drop.add(b)
        elif pb < pa:
            to_drop.add(a)
        else:
            to_drop.add(b)
    return sorted(to_drop)


def group_correlation_subsets(df: pd.DataFrame, groups: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """§8: within-group Pearson correlation."""
    out: Dict[str, pd.DataFrame] = {}
    for gname, feats in groups.items():
        present = [f for f in feats if f in df.columns]
        if len(present) < 2:
            continue
        sub = df[present]
        out[gname] = sub.corr(method="pearson", numeric_only=True)
    return out


def build_final_feature_set(
    available: Set[str],
    corr_pairs: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    §9: Rule-based final selection + drop redundant from high correlation.
    """
    picked: List[str] = []
    notes: List[str] = []

    # Scale: keep 1–2 log variables (prefer log_n_unique_voters, log_total_votes)
    for c in ("log_n_unique_voters", "log_total_votes"):
        if c in available:
            picked.append(c)
    if not any(x in available for x in ("log_n_unique_voters", "log_total_votes")):
        for c in ("log_n_proposals", "log_mean_sum_vp"):
            if c in available and len(picked) < 2:
                picked.append(c)

    # Participation: robust mean + std only
    for c in ("mean_robust_participation_vp", "std_robust_participation_vp"):
        if c in available:
            picked.append(c)

    # Governance: gini OR hhi (prefer gini if both)
    if "gini_voting_power" in available:
        picked.append("gini_voting_power")
        notes.append("Governance: kept gini_voting_power (dropped hhi as duplicate concept).")
    elif "hhi_voting_power" in available:
        picked.append("hhi_voting_power")

    if "whale_ratio_top1pct" in available:
        picked.append("whale_ratio_top1pct")

    # Activity: both
    for c in ACTIVITY_FEATURES:
        if c in available:
            picked.append(c)

    # Representative: z_rep behavioural only
    z_beh = []
    for suf in Z_REP_BEHAVIOUR_SUFFIXES:
        name = f"z_rep_{suf}"
        if name in available:
            z_beh.append(name)
    picked.extend(z_beh)

    # Dedupe while preserving order
    seen = set()
    final_ordered = []
    for c in picked:
        if c not in seen:
            seen.add(c)
            final_ordered.append(c)

    # Remove any that are in redundant drop list from correlation (among picked)
    priority = (
        SCALE_FEATURES
        + [f"log_{x}" for x in LOG_SCALE_TARGETS]
        + ["mean_robust_participation_vp", "std_robust_participation_vp"]
        + GOVERNANCE_FEATURES
        + ACTIVITY_FEATURES
        + [f"z_rep_{s}" for s in Z_REP_BEHAVIOUR_SUFFIXES]
    )
    redundant = set(mark_redundant_from_pairs(corr_pairs, priority))
    final_filtered = [c for c in final_ordered if c not in redundant]
    return final_filtered, notes


def plot_heatmap(corr: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(corr.columns) * 0.35), max(8, len(corr.columns) * 0.35)))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"C:\Users\DELL\Desktop\Honda_doc\draft paper\dao_feature_table.xlsx",
        help="Excel or CSV path (demand default: dao_feature_table.xlsx).",
    )
    parser.add_argument(
        "--fallback-csv",
        default="",
        help="If primary input missing, use this CSV path.",
    )
    parser.add_argument("--out-dir", default="outputs/feature_screening")
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = Path(args.input)
    df: Optional[pd.DataFrame] = None
    if primary.exists():
        try:
            df = load_input(primary)
            log(f"Loaded: {primary}")
        except ImportError as exc:
            warn(str(exc))

    if df is None:
        if args.fallback_csv:
            fb = Path(args.fallback_csv)
            if not fb.is_absolute():
                fb = base / fb
            if fb.exists():
                df = load_input(fb)
                warn(f"Used explicit fallback CSV: {fb}")
            else:
                raise FileNotFoundError(f"Fallback CSV not found: {fb}")
        else:
            fb = base.parent / "scripts" / "data" / "processed" / "dao_feature_table.csv"
            if fb.exists():
                df = load_input(fb)
                warn(f"Used default fallback CSV: {fb}")
            else:
                raise FileNotFoundError(
                    f"Could not load data. Install openpyxl for Excel, or pass --fallback-csv. "
                    f"Primary path was: {primary}"
                )

    log(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # STEP 1
    miss_before = report_missing(df.drop(columns=["space"], errors="ignore"))
    miss_before.to_csv(out_dir / "missing_value_counts_before.csv", header=["missing_count"])
    print("\n=== Missing values (before, columns with any NA) ===")
    print(miss_before.to_string() if len(miss_before) else "(none)")

    df_id, df_num = separate_id_and_numeric(df)
    original_feature_count = df_num.shape[1]

    df_clean, dropped_missing, dropped_const = basic_cleaning(df_num)

    # STEP 2 preferences (before log)
    df_clean, removed_pref = apply_group_preferences(df_clean)
    log(f"Removed by preference (robust/z over raw): {removed_pref}")

    # STEP 3 log
    df_with_log, log_created = add_log_features(df_clean)

    # Prefer log versions for scale: drop raw scale duplicates if log exists
    dropped_for_log: List[str] = []
    for base in LOG_SCALE_TARGETS:
        logcol = f"log_{base}"
        if logcol in df_with_log.columns and base in df_with_log.columns:
            df_with_log = df_with_log.drop(columns=[base])
            dropped_for_log.append(base)

    # STEP 4 winsorize
    df_wins, winsor_affected = winsorize_df(df_with_log)

    # STEP 5 scale (for clustering matrix; correlation on pre-scaled numeric is still Pearson on raw processed)
    df_scaled, _ = robust_scale(df_wins)

    # STEP 6 correlation (Pearson on processed-but-not-robust-scaled features — demand says correlation matrix;
    #  typically computed on same scale as analysis: use winsorized processed features)
    corr_full = df_wins.corr(method="pearson", numeric_only=True)
    corr_full.to_csv(out_dir / "correlation_matrix_full.csv")
    plot_heatmap(corr_full, out_dir / "heatmap_full.png", "Full correlation (processed)")

    # STEP 7 high correlation
    pairs = correlation_pairs(corr_full, threshold=args.corr_threshold)
    pairs.to_csv(out_dir / "high_correlation_pairs.csv", index=False)

    redundant_auto = mark_redundant_from_pairs(
        pairs,
        list(df_wins.columns),
    )

    # STEP 8 group-wise
    all_cols = set(df_wins.columns)
    groups_map = {
        "A_scale": [c for c in SCALE_FEATURES + [f"log_{x}" for x in LOG_SCALE_TARGETS] if c in all_cols],
        "B_participation": [c for c in PARTICIPATION_FEATURES if c in all_cols],
        "C_governance": [c for c in GOVERNANCE_FEATURES if c in all_cols],
        "D_activity": [c for c in ACTIVITY_FEATURES if c in all_cols],
        "E_raw_rep": find_columns_by_prefix(df_wins.columns, "raw_rep_"),
        "F_z_rep": find_columns_by_prefix(df_wins.columns, "z_rep_"),
    }
    group_corrs = group_correlation_subsets(df_wins, groups_map)
    for gname, gcorr in group_corrs.items():
        gcorr.to_csv(out_dir / f"correlation_within_group_{gname}.csv")
        print(f"\n=== Within-group summary: {gname} ===")
        print(f"Features: {list(gcorr.columns)}")
        # upper triangle mean abs corr
        gc = gcorr.values
        tri = []
        for i in range(len(gc)):
            for j in range(i + 1, len(gc)):
                tri.append(abs(gc[i, j]))
        if tri:
            print(f"Mean |r| (off-diagonal): {float(np.mean(tri)):.4f}")

    # STEP 9 final set
    final_features, interpret_notes = build_final_feature_set(set(df_wins.columns), pairs)
    # also remove redundant from auto list that are not in final rule list
    final_set = [c for c in final_features if c in df_wins.columns]
    dropped_redundant = [c for c in redundant_auto if c not in final_set]

    # Save scaled matrix for downstream clustering
    df_scaled.index = df_id["space"].values
    scaled_out = pd.concat([df_id.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
    scaled_out.to_csv(out_dir / "df_scaled.csv", index=False)
    try:
        scaled_out.to_parquet(out_dir / "df_scaled.parquet", index=False)
    except Exception as exc:
        warn(f"Parquet export skipped ({exc}); CSV retained.")

    # Reduced heatmap on final features
    if final_set:
        corr_reduced = df_wins[final_set].corr(method="pearson", numeric_only=True)
        corr_reduced.to_csv(out_dir / "correlation_matrix_reduced.csv")
        plot_heatmap(corr_reduced, out_dir / "heatmap_reduced.png", "Reduced feature correlation")

    pd.Series(final_set, name="feature").to_csv(out_dir / "final_selected_features.csv", index=False)
    pd.Series(dropped_redundant, name="feature").to_csv(out_dir / "dropped_redundant_features.csv", index=False)
    all_dropped = (
        [f"high_missing:{c}" for c in dropped_missing]
        + [f"constant:{c}" for c in dropped_const]
        + [f"prefer_robust_or_z:{c}" for c in removed_pref]
        + [f"replaced_by_log:{c}" for c in dropped_for_log]
        + [f"high_corr:{c}" for c in redundant_auto]
    )
    pd.Series(all_dropped, name="drop_reason").to_csv(out_dir / "dropped_features_all_steps.csv", index=False)

    with open(out_dir / "winsorized_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(winsor_affected))

    # STEP 11 report
    print("\n=== PIPELINE REPORT ===")
    print(f"Original numeric feature count: {original_feature_count}")
    removed_total = (
        len(dropped_missing)
        + len(dropped_const)
        + len(removed_pref)
        + len(dropped_for_log)
        + len(set(redundant_auto))
    )
    print(f"Approx. removed / transformed (see dropped_features_all_steps.csv): {removed_total}")
    print(f"Final feature count: {len(final_set)}")
    print("Final feature names:")
    for name in final_set:
        print(f"  - {name}")
    print("\nDimensions captured:")
    print("  - Scale / size: log-transformed voter/vote/proposal proxies (1-2 features).")
    print("  - Participation: robust VP participation (mean + std).")
    print("  - Governance: concentration (Gini or HHI) + whale share.")
    print("  - Activity: proposal frequency + repeat voter behaviour.")
    print("  - Representative behaviour: z-scored voter-centroid preferences (for/against/abstain, entropy, alignment).")
    for n in interpret_notes:
        print(f"  Note: {n}")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
