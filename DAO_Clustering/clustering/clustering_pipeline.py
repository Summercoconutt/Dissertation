"""
DAO clustering pipeline (dissertation-grade).
Run end-to-end: load -> scale -> KMeans -> profiles -> plots -> hierarchical -> GMM -> interpretation.

Outputs: ./outputs/clustering_pipeline/ (relative to this script's directory)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_INIT = 25

CLUSTERING_FEATURES: List[str] = [
    "log_n_unique_voters",
    "mean_robust_participation_vp",
    "std_robust_participation_vp",
    "gini_voting_power",
    "whale_ratio_top1pct",
    "proposal_frequency_per_30d",
    "repeat_voter_rate",
    "z_rep_pct_for_votes",
    "z_rep_pct_against_votes",
    "z_rep_pct_abstain_votes",
    "z_rep_choice_entropy",
    "z_rep_pct_aligned_with_majority",
]

# Manual override: set to None to use automatic recommendation
BEST_K_OVERRIDE: Optional[int] = None


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def ensure_out_dir(base: Path) -> Path:
    out = base / "outputs" / "clustering_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported: {path}")


def step1_load_and_prepare(
    df: pd.DataFrame, required_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if "space" not in df.columns:
        raise ValueError("Column 'space' is required.")
    df = df.copy()
    # Derive log_n_unique_voters if raw scale column exists but log is absent
    if "log_n_unique_voters" not in df.columns and "n_unique_voters" in df.columns:
        nv = pd.to_numeric(df["n_unique_voters"], errors="coerce").clip(lower=0)
        df["log_n_unique_voters"] = np.log1p(nv)
        log("Derived log_n_unique_voters from n_unique_voters (log1p).")

    available = [c for c in required_features if c in df.columns]
    missing = [c for c in required_features if c not in df.columns]
    for m in missing:
        warn(f"Missing column (skipped): {m}")
    if not available:
        raise ValueError("No clustering features available after column check.")

    df_meta = df[["space"]].copy()
    df_features = df[available].apply(pd.to_numeric, errors="coerce")
    miss = df_features.isna().sum()
    print("\n=== Missing counts per selected feature ===")
    print(miss.to_string())
    for c in df_features.columns:
        if df_features[c].isna().any():
            med = df_features[c].median()
            df_features[c] = df_features[c].fillna(med)
    return df_meta, df_features, available


def step2_scale(df_features: pd.DataFrame) -> Tuple[np.ndarray, RobustScaler, pd.DataFrame]:
    scaler = RobustScaler()
    X = scaler.fit_transform(df_features.to_numpy(dtype=float))
    scaled_df = pd.DataFrame(X, columns=df_features.columns, index=df_features.index)
    print("\n=== Before scaling (original features) ===")
    print(df_features.describe().T.to_string())
    print("\n=== After RobustScaler ===")
    print(scaled_df.describe().T.to_string())
    return X, scaler, scaled_df


def step3_kmeans_eval(X: np.ndarray, k_min: int, k_max: int) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        row = {
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
        }
        rows.append(row)
        log(f"KMeans k={k}: silhouette={row['silhouette']:.4f}, CH={row['calinski_harabasz']:.2f}, DB={row['davies_bouldin']:.4f}")
    return pd.DataFrame(rows)


def recommend_k(eval_df: pd.DataFrame) -> int:
    """Higher silhouette & CH, lower DB; mild penalty for large k."""
    df = eval_df.copy()
    # Min-max to [0,1], higher is better for all in composite
    def mm(series: pd.Series, higher_better: bool) -> pd.Series:
        s = series.astype(float)
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-12:
            return pd.Series(0.5, index=s.index)
        if higher_better:
            return (s - lo) / (hi - lo)
        return (hi - s) / (hi - lo)  # invert for lower-better

    sil = mm(df["silhouette"], True)
    ch = mm(df["calinski_harabasz"], True)
    db = mm(df["davies_bouldin"], False)
    frag_penalty = (df["k"] - df["k"].min()) * 0.03
    composite = sil + ch + db - frag_penalty
    best_idx = int(composite.idxmax())
    k_star = int(df.loc[best_idx, "k"])
    return k_star


def plot_metric_curves(eval_df: pd.DataFrame, out_dir: Path) -> None:
    ks = eval_df["k"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, eval_df["inertia"], "o-", color="C0")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow curve (inertia vs k)")
    ax.set_xticks(ks)
    plt.tight_layout()
    plt.savefig(out_dir / "elbow_curve.png", dpi=200)
    plt.close()

    for col, fname, title in [
        ("silhouette", "silhouette_curve.png", "Silhouette score vs k"),
        ("calinski_harabasz", "calinski_harabasz_curve.png", "Calinski-Harabasz vs k"),
        ("davies_bouldin", "davies_bouldin_curve.png", "Davies-Bouldin vs k (lower is better)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, eval_df[col], "o-", color="C1")
        ax.set_xlabel("k")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.set_xticks(ks)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()


def step6_profiles(
    df_assign: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw mean/median/std per cluster + standardized (z of cluster mean vs global mean)."""
    g = df_assign.groupby("cluster", observed=True)
    mean_prof = g[feature_cols].mean()
    med_prof = g[feature_cols].median()
    std_prof = g[feature_cols].std(ddof=0)
    sizes = g.size().rename("n_daos")
    mean_prof = mean_prof.join(sizes)

    global_mean = df_assign[feature_cols].mean()
    global_std = df_assign[feature_cols].std(ddof=0).replace(0, np.nan)
    z_rows = []
    for cl in mean_prof.index:
        row = (mean_prof.loc[cl, feature_cols] - global_mean) / global_std
        z_rows.append(row)
    std_profile = pd.DataFrame(z_rows, index=mean_prof.index)
    return mean_prof, med_prof, std_prof, std_profile


def plot_cluster_heatmaps(mean_prof: pd.DataFrame, std_profile: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> None:
    plot_mean = mean_prof[feature_cols]
    plt.figure(figsize=(max(10, len(feature_cols) * 0.6), max(4, len(plot_mean) * 0.8)))
    sns.heatmap(plot_mean, annot=True, fmt=".3f", cmap="RdYlBu_r", center=plot_mean.values.mean())
    plt.title("Cluster mean profiles (original feature scale)")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_profile_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(10, len(feature_cols) * 0.6), max(4, len(std_profile) * 0.8)))
    sns.heatmap(std_profile, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Standardized cluster profiles (z vs global mean)")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_profile_heatmap_standardized.png", dpi=200)
    plt.close()


def plot_boxplots(df_assign: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> None:
    n = len(feature_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.boxplot(data=df_assign, x="cluster", y=col, ax=ax)
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", rotation=0)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Features by cluster", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_feature_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_cluster_sizes(df_assign: pd.DataFrame, out_dir: Path) -> None:
    counts = df_assign["cluster"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color="steelblue")
    plt.xlabel("Cluster")
    plt.ylabel("Number of DAOs")
    plt.title("Cluster sizes")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_sizes.png", dpi=200)
    plt.close()


def plot_cluster_centers_radar(km: KMeans, feature_cols: List[str], scaler: RobustScaler, out_dir: Path) -> None:
    """Approximate: inverse-transform KMeans centers to original-ish scale for bar chart."""
    centers = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers)
    df_c = pd.DataFrame(centers_orig, columns=feature_cols)
    df_c.index = [f"C{i}" for i in range(len(df_c))]
    df_c.T.plot(kind="bar", figsize=(max(12, len(feature_cols) * 0.5), 5), width=0.85)
    plt.title("KMeans cluster centers (inverse RobustScaler)")
    plt.xlabel("Feature")
    plt.ylabel("Approx. original scale")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_centers_grouped_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


def step8_pca(X: np.ndarray, labels: np.ndarray, spaces: pd.Series, out_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = pca.fit_transform(X)
    print("\n=== PCA explained variance ratio ===")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}, PC2: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"Total: {pca.explained_variance_ratio_.sum():.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", alpha=0.75, s=28)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("DAO positions (PCA), colored by KMeans cluster")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_clusters.png", dpi=200)
    plt.close()

    pd.DataFrame({"space": spaces.values, "pc1": emb[:, 0], "pc2": emb[:, 1], "cluster": labels}).to_csv(
        out_dir / "pca_coordinates.csv", index=False
    )


def step9_hierarchical(X: np.ndarray, best_k: int, kmeans_labels: np.ndarray, spaces: pd.Series, out_dir: Path) -> float:
    Z = linkage(X, method="ward")
    plt.figure(figsize=(14, 6))
    dendrogram(Z, no_labels=True, color_threshold=0)
    plt.title("Hierarchical clustering dendrogram (Ward)")
    plt.xlabel("DAO index (leaf order)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(out_dir / "hierarchical_dendrogram.png", dpi=200)
    plt.close()

    hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_labels = hc.fit_predict(X)
    ari = float(adjusted_rand_score(kmeans_labels, hc_labels))
    comp = pd.DataFrame({"space": spaces.values, "kmeans_cluster": kmeans_labels, "hierarchical_cluster": hc_labels})
    comp.to_csv(out_dir / "hierarchical_vs_kmeans_comparison.csv", index=False)
    log(f"Adjusted Rand Index (KMeans vs hierarchical): {ari:.4f}")
    return ari


def step10_gmm(X: np.ndarray, k_min: int, k_max: int, out_dir: Path) -> None:
    rows = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=5)
        gmm.fit(X)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        rows.append({"k": k, "bic": bic, "aic": aic})
        log(f"GMM k={k}: BIC={bic:.2f}, AIC={aic:.2f}")
    ev = pd.DataFrame(rows)
    ev.to_csv(out_dir / "gmm_model_selection.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ev["k"], ev["bic"], "o-", label="BIC")
    ax.plot(ev["k"], ev["aic"], "s-", label="AIC")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title("GMM model selection (lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "gmm_bic_aic.png", dpi=200)
    plt.close()


def step11_interpretation(mean_prof: pd.DataFrame, feature_cols: List[str], out_path: Path) -> str:
    """High/low vs weighted global reference per feature, per cluster."""
    lines = []
    # Reference: weighted mean of cluster means by cluster size
    n_col = "n_daos" if "n_daos" in mean_prof.columns else None
    if n_col:
        w = mean_prof[n_col]
        ref = (mean_prof[feature_cols].T * w).T.sum() / w.sum()
    else:
        ref = mean_prof[feature_cols].mean()

    labels_map = {
        "mean_robust_participation_vp": "participation (robust VP)",
        "std_robust_participation_vp": "participation variability",
        "gini_voting_power": "inequality (Gini)",
        "whale_ratio_top1pct": "whale concentration",
        "proposal_frequency_per_30d": "proposal activity",
        "repeat_voter_rate": "repeat voting",
        "z_rep_pct_for_votes": "behaviour: FOR tendency",
        "z_rep_pct_against_votes": "behaviour: AGAINST tendency",
        "z_rep_pct_abstain_votes": "behaviour: ABSTAIN tendency",
        "z_rep_choice_entropy": "choice entropy",
        "z_rep_pct_aligned_with_majority": "alignment with majority",
        "log_n_unique_voters": "scale (log unique voters)",
    }

    for cl in mean_prof.index:
        parts = [f"Cluster {int(cl)} (n={int(mean_prof.loc[cl, n_col]) if n_col else '?'}):"]
        for f in feature_cols:
            v = float(mean_prof.loc[cl, f])
            r = float(ref[f])
            tag = "high" if v > r else "low"
            name = labels_map.get(f, f)
            parts.append(f"  {tag} {name} (mean={v:.4f} vs ref={r:.4f})")
        lines.append("\n".join(parts))
        lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print("\n=== Cluster interpretation (draft) ===\n")
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="DAO clustering pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"C:\Users\DELL\Desktop\Honda_doc\draft paper\dao_feature_table.xlsx"),
    )
    parser.add_argument(
        "--fallback-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "scripts" / "data" / "processed" / "dao_feature_table.csv",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = ensure_out_dir(base)

    inp = args.input
    if inp.exists():
        df = load_table(inp)
        log(f"Loaded: {inp}")
    elif args.fallback_csv.exists():
        df = load_table(args.fallback_csv)
        warn(f"Primary input missing; using fallback: {args.fallback_csv}")
    else:
        raise FileNotFoundError(f"No input found: {inp}")

    log(f"Rows: {len(df)}, columns: {len(df.columns)}")

    df_meta, df_features, used_features = step1_load_and_prepare(df, CLUSTERING_FEATURES)
    X, scaler, scaled_df = step2_scale(df_features)
    scaled_out = pd.concat([df_meta.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
    scaled_out.to_csv(out_dir / "scaled_features.csv", index=False)

    eval_df = step3_kmeans_eval(X, args.k_min, args.k_max)
    eval_df.to_csv(out_dir / "clustering_evaluation_k2_to_k6.csv", index=False)
    plot_metric_curves(eval_df, out_dir)

    recommended_k = recommend_k(eval_df)
    best_k = BEST_K_OVERRIDE if BEST_K_OVERRIDE is not None else recommended_k
    print(f"\n>>> Recommended k (automatic): {recommended_k}")
    if BEST_K_OVERRIDE is not None:
        print(f">>> Using manual BEST_K_OVERRIDE: {best_k}")
    else:
        print(f">>> Using BEST_K = {best_k}")

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km_final.fit_predict(X)

    df_assign = pd.concat([df_meta.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
    df_assign["cluster"] = labels
    df_assign.to_csv(out_dir / "final_cluster_assignments.csv", index=False)

    mean_prof, med_prof, std_prof, std_profile = step6_profiles(df_assign, used_features)
    mean_prof.to_csv(out_dir / "cluster_profile_mean.csv")
    med_prof.to_csv(out_dir / "cluster_profile_median.csv")
    std_prof.to_csv(out_dir / "cluster_profile_std.csv")
    std_profile.to_csv(out_dir / "cluster_profile_standardized.csv")

    plot_cluster_heatmaps(mean_prof, std_profile, used_features, out_dir)
    plot_boxplots(df_assign, used_features, out_dir)
    plot_cluster_sizes(df_assign, out_dir)
    plot_cluster_centers_radar(km_final, used_features, scaler, out_dir)

    step8_pca(X, labels, df_meta["space"], out_dir)

    ari = step9_hierarchical(X, best_k, labels, df_meta["space"], out_dir)

    step10_gmm(X, args.k_min, args.k_max, out_dir)

    step11_interpretation(mean_prof, used_features, out_dir / "cluster_interpretation.txt")

    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    print(f"Number of DAOs: {len(df_assign)}")
    print(f"Number of features used: {len(used_features)}")
    print(f"Features: {used_features}")
    print(f"Recommended k: {recommended_k}; BEST_K used: {best_k}")
    print("\nKMeans evaluation (all k):")
    print(eval_df.to_string(index=False))
    print(f"\nAdjusted Rand Index (KMeans vs hierarchical, k={best_k}): {ari:.4f}")
    print(
        "\nStability note: Higher ARI suggests hierarchical structure aligns with KMeans partitions; "
        "interpret clusters using profiles and domain knowledge."
    )
    print(f"\nAll outputs: {out_dir}")


if __name__ == "__main__":
    main()
