from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import log


def detect_scale_driven_features(cols: list[str]) -> list[str]:
    keys = (
        "n_",
        "total",
        "avg_voting_power",
        "median_voting_power",
        "max_voting_power",
        "sum_vp",
        "sum_",
    )
    return [c for c in cols if any(k in c for k in keys)]


def cap_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        low = q1 - 3.0 * iqr
        high = q3 + 3.0 * iqr
        out[c] = s.clip(lower=low, upper=high)
    return out


def apply_log_transform(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    transformed = []
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        if s.dropna().empty:
            continue
        skew = s.dropna().skew()
        if pd.notna(skew) and skew > 1.0 and (s.dropna() >= 0).all():
            out[c] = np.log1p(s)
            transformed.append(c)
    return out, transformed


def correlation_screening(corr: pd.DataFrame, threshold: float = 0.9) -> tuple[list[str], list[tuple[str, str, float]]]:
    cols = list(corr.columns)
    keep: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for c in cols:
        drop = False
        for k in keep:
            v = corr.loc[c, k]
            if pd.notna(v) and abs(v) >= threshold:
                dropped.append((c, k, float(v)))
                drop = True
                break
        if not drop:
            keep.append(c)
    return keep, dropped


def save_heatmap(corr: pd.DataFrame, out_png: Path) -> None:
    if corr.empty:
        return
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.title("DAO Feature Correlation Heatmap")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=r"data\processed\dao_feature_table.parquet")
    parser.add_argument("--out-matrix", default=r"data\processed\dao_feature_matrix.parquet")
    parser.add_argument("--out-selected", default=r"outputs\tables\selected_features.csv")
    parser.add_argument("--out-corr", default=r"outputs\tables\correlation_matrix.csv")
    parser.add_argument("--out-dropped", default=r"outputs\tables\dropped_correlated_features.csv")
    parser.add_argument("--out-metadata", default=r"outputs\tables\feature_metadata.csv")
    parser.add_argument("--out-heatmap", default=r"outputs\figures\correlation_heatmap.png")
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.in_path))
    if "space" not in df.columns:
        raise ValueError("Input dao_feature_table must contain `space` column.")

    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != "space"]
    x = df[["space"] + numeric_cols].copy()

    # Missing values
    x[numeric_cols] = x[numeric_cols].apply(pd.to_numeric, errors="coerce")
    x[numeric_cols] = x[numeric_cols].fillna(x[numeric_cols].median(numeric_only=True))

    # Outlier capping
    x = cap_outliers_iqr(x, numeric_cols)

    # Log-transform heavy tails
    x, log_cols = apply_log_transform(x, numeric_cols)

    # Standardize
    scaler = StandardScaler()
    z = x.copy()
    if numeric_cols:
        z[numeric_cols] = scaler.fit_transform(z[numeric_cols])

    # Correlation screening
    corr = z[numeric_cols].corr(numeric_only=True) if numeric_cols else pd.DataFrame()
    selected_cols, dropped = correlation_screening(corr, threshold=args.corr_threshold) if not corr.empty else ([], [])

    # Feature metadata
    scale_driven = detect_scale_driven_features(numeric_cols)
    meta = pd.DataFrame(
        {
            "feature": numeric_cols,
            "is_scale_driven": [c in scale_driven for c in numeric_cols],
            "log_transformed": [c in log_cols for c in numeric_cols],
            "selected_for_clustering": [c in selected_cols for c in numeric_cols],
        }
    )

    # Save outputs
    out_matrix = Path(args.out_matrix)
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    z_out = z[["space"] + selected_cols] if selected_cols else z[["space"]]
    z_out.to_parquet(out_matrix, index=False)
    z_out.to_csv(out_matrix.with_suffix(".csv"), index=False)

    Path(args.out_selected).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": selected_cols}).to_csv(args.out_selected, index=False)
    corr.to_csv(args.out_corr)
    pd.DataFrame(dropped, columns=["dropped_feature", "kept_feature", "correlation"]).to_csv(args.out_dropped, index=False)
    meta.to_csv(args.out_metadata, index=False)
    save_heatmap(corr, Path(args.out_heatmap))

    log(f"Saved cleaned+standardized feature matrix: {out_matrix}")
    log(f"Saved selected features: {args.out_selected}")
    log(f"Saved correlation matrix: {args.out_corr}")
    log(f"Saved dropped correlated features: {args.out_dropped}")
    log(f"Saved feature metadata: {args.out_metadata}")
    log(f"Saved correlation heatmap: {args.out_heatmap}")


if __name__ == "__main__":
    main()

