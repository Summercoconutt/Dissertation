"""
DAO-level clustering validation & debugging pipeline (dissertation).
Tests whether meaningful cluster structure exists vs continuous / outlier-dominated structure.

Edit CONFIGURATION below, then run: python clustering_validation_pipeline.py
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURATION — edit these
# =============================================================================

INPUT_PATH = Path(
    r"C:\Users\DELL\Desktop\Honda_doc\draft paper\dao_feature_table.xlsx"
)
# Fallback if INPUT_PATH missing
FALLBACK_CSV = Path(__file__).resolve().parent.parent / "DAO_Clustering" / "scripts" / "data" / "processed" / "dao_feature_table.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "clustering_validation"
# Step 7-style bundle (heatmap, distributions, assignments, PCA): mirrors dissertation demands
CLUSTER_RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "cluster_results"

ID_COLUMN = "space"

# Numeric features used for clustering diagnostics (edit freely)
FEATURE_COLUMNS: List[str] = [
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

# Columns to apply log1p (must be non-negative after shift if needed)
LOG1P_COLUMNS: List[str] = [
    "proposal_frequency_per_30d",
]

RANDOM_STATE = 42
N_INIT = 25
K_RANGE = [2, 3, 4, 5, 6]
CORR_THRESHOLD = 0.8
Z_OUTLIER = 3.0
IQR_K = 1.5
# Flag DAO if z>|Z_OUTLIER| in ANY feature OR outside IQR in >= 2 features
IQR_COUNT_FLAG = 2

DBSCAN_EPS_GRID = [0.5, 1.0, 1.5, 2.0, 2.5]
DBSCAN_MIN_SAMPLES_GRID = [3, 5, 8]

NULL_N_PERM = 20
STABILITY_SEEDS = list(range(RANDOM_STATE, RANDOM_STATE + 10))


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


# =============================================================================
# 1. Load
# =============================================================================


def load_data() -> pd.DataFrame:
    if INPUT_PATH.exists():
        if INPUT_PATH.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(INPUT_PATH)
        return pd.read_csv(INPUT_PATH)
    if FALLBACK_CSV.exists():
        warn(f"INPUT_PATH not found; using fallback: {FALLBACK_CSV}")
        return pd.read_csv(FALLBACK_CSV)
    raise FileNotFoundError(f"No input: {INPUT_PATH}")


def ensure_log_n_unique_voters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_n_unique_voters" not in out.columns and "n_unique_voters" in out.columns:
        out["log_n_unique_voters"] = np.log1p(
            pd.to_numeric(out["n_unique_voters"], errors="coerce").clip(lower=0)
        )
        log("Derived log_n_unique_voters from n_unique_voters.")
    return out


# =============================================================================
# 2. Preprocess
# =============================================================================


def preprocess_data(
    df: pd.DataFrame, feature_cols: List[str], log1p_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, RobustScaler, List[str]]:
    """Returns (df_ids, X_original_used, scaler_fitted_on_train, used_cols)."""
    df = ensure_log_n_unique_voters(df)
    missing = [c for c in feature_cols if c not in df.columns]
    for m in missing:
        warn(f"Feature not in data, skipped: {m}")
    use = [c for c in feature_cols if c in df.columns]
    if not use:
        raise ValueError("No valid feature columns.")

    sub = df[[ID_COLUMN] + use].copy() if ID_COLUMN in df.columns else df[use].copy()
    if ID_COLUMN not in sub.columns:
        sub.insert(0, ID_COLUMN, range(len(sub)))

    for c in use:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    for c in log1p_cols:
        if c in sub.columns:
            sub[c] = np.log1p(sub[c].clip(lower=0))

    before = len(sub)
    sub = sub.dropna(subset=use)
    dropped = before - len(sub)
    log(f"Dropped {dropped} rows with missing values in selected features (before: {before}).")

    X = sub[use].to_numpy(dtype=float)
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    return sub[[ID_COLUMN]].reset_index(drop=True), pd.DataFrame(X, columns=use), scaler, use


# =============================================================================
# 3. Feature distributions
# =============================================================================


def plot_feature_distributions(df_feat: pd.DataFrame, feat_cols: List[str], out_dir: Path) -> pd.DataFrame:
    dist_dir = out_dir / "feature_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in feat_cols:
        s = df_feat[c].dropna()
        skew = float(stats.skew(s)) if len(s) > 2 else np.nan
        kurt = float(stats.kurtosis(s)) if len(s) > 4 else np.nan
        heavy = "likely heavy-tailed" if (abs(skew) > 1.0 or abs(kurt) > 3) else "moderate"
        rows.append(
            {
                "feature": c,
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
                "skewness": skew,
                "kurtosis": kurt,
                "skew_note": heavy,
            }
        )
        plt.figure(figsize=(7, 4))
        sns.histplot(s, kde=True, stat="density")
        plt.title(f"{c} (skew={skew:.3f})")
        plt.tight_layout()
        plt.savefig(dist_dir / f"hist_{c}.png", dpi=160)
        plt.close()
    return pd.DataFrame(rows)


# =============================================================================
# 4. Correlation
# =============================================================================


def compute_correlation_diagnostics(df_feat: pd.DataFrame, feat_cols: List[str], out_dir: Path) -> pd.DataFrame:
    corr = df_feat[feat_cols].corr(method="pearson")
    corr.to_csv(out_dir / "correlation_matrix.csv")
    plt.figure(figsize=(max(8, len(feat_cols) * 0.35), max(6, len(feat_cols) * 0.35)))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) > CORR_THRESHOLD:
                pairs.append({"feature_a": a, "feature_b": b, "correlation": v})
    hp = pd.DataFrame(pairs)
    hp.to_csv(out_dir / "high_corr_pairs.csv", index=False)
    if len(hp) > len(feat_cols):
        warn(f"Many highly correlated pairs ({len(hp)}); multicollinearity may affect clustering stability.")
    return hp


# =============================================================================
# 5. Outliers
# =============================================================================


def detect_outliers(
    df_ids: pd.DataFrame, df_feat: pd.DataFrame, feat_cols: List[str], out_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (report, df_all_with_flags, df_no_outliers)."""
    z_flags = []
    iqr_flags = []
    rows = []
    for idx in range(len(df_feat)):
        z_any = False
        iqr_hits = 0
        detail = {ID_COLUMN: df_ids.iloc[idx, 0]}
        for c in feat_cols:
            col = df_feat[c]
            s = col.dropna()
            mu, sig = s.mean(), s.std()
            if sig and sig > 0:
                z = abs((df_feat.iloc[idx][c] - mu) / sig)
            else:
                z = 0.0
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
            v = df_feat.iloc[idx][c]
            outside = v < lo or v > hi
            if z > Z_OUTLIER:
                z_any = True
            if outside:
                iqr_hits += 1
            detail[f"z_{c}"] = z
            detail[f"iqr_out_{c}"] = outside
        flag = z_any or (iqr_hits >= IQR_COUNT_FLAG)
        detail["outlier_flag"] = flag
        detail["iqr_extreme_count"] = iqr_hits
        rows.append(detail)

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "outlier_report.csv", index=False)
    bad = report["outlier_flag"]
    n_out = int(bad.sum())
    log(f"Outliers flagged: {n_out} / {len(report)}")
    if n_out:
        names = report.loc[bad, ID_COLUMN].astype(str).tolist()
        safe = ", ".join(names[:50])
        print("Outlier DAOs (first 50):", safe.encode("ascii", errors="replace").decode("ascii"))

    df_all = pd.concat([df_ids.reset_index(drop=True), df_feat.reset_index(drop=True), report[["outlier_flag"]]], axis=1)
    df_clean = df_all.loc[~bad].reset_index(drop=True)
    return report, df_all, df_clean


# =============================================================================
# 6. PCA
# =============================================================================


def _plot_pca_eigenvector_bars(
    pca: PCA, feat_cols: List[str], out_dir: Path, tag: str, n_plot: int = 3
) -> None:
    """Bar chart of PCA loadings (eigenvector coefficients) per component — interpret PC direction."""
    n_plot = min(n_plot, pca.components_.shape[0])
    x = np.arange(len(feat_cols))
    for i in range(n_plot):
        load = pca.components_[i]
        plt.figure(figsize=(max(9, len(feat_cols) * 0.45), 5))
        plt.bar(x, load, color=["#2c7fb8" if load[j] >= 0 else "#d95f0e" for j in range(len(feat_cols))])
        plt.xticks(x, feat_cols, rotation=45, ha="right", fontsize=8)
        plt.ylabel("Value")
        plt.xlabel("Feature")
        plt.title(f"PCA Eigenvector {i + 1}")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(out_dir / f"pca_eigenvector_PC{i + 1}_{tag}.png", dpi=200)
        plt.close()


def run_pca_scaled(
    X_scaled: np.ndarray, ids: pd.Series, feat_cols: List[str], tag: str, out_dir: Path
) -> None:
    n_comp = min(10, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_scaled)
    ev = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    ev.to_csv(out_dir / f"pca_explained_variance_{tag}.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(ev) + 1), ev["cumulative"], "o-")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA cumulative variance ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"pca_cumulative_variance_{tag}.png", dpi=180)
    plt.close()

    n_pc_csv = min(3, n_comp)
    load_cols = [f"PC{i+1}_loading" for i in range(n_pc_csv)]
    loadings = pd.DataFrame(pca.components_[:n_pc_csv].T, index=feat_cols, columns=load_cols)
    loadings.to_csv(out_dir / f"pca_loadings_{tag}.csv")

    _plot_pca_eigenvector_bars(pca, feat_cols, out_dir, tag, n_plot=3)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.5, s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA scatter ({tag})")
    # annotate extremes
    r = np.sum(Z[:, :2] ** 2, axis=1).argsort()
    for j in list(r[-3:]) + list(r[:3]):
        plt.annotate(str(ids.iloc[j])[:20], (Z[j, 0], Z[j, 1]), fontsize=6, alpha=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / f"pca_scatter_{tag}.png", dpi=180)
    plt.close()


# =============================================================================
# 7. K-means validation
# =============================================================================


def run_kmeans_validation(
    X_scaled: np.ndarray,
    X_orig: np.ndarray,
    df_ids: pd.DataFrame,
    feat_cols: List[str],
    scaler: RobustScaler,
    tag: str,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        lab = km.fit_predict(X_scaled)
        row = {
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, lab),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, lab),
            "davies_bouldin": davies_bouldin_score(X_scaled, lab),
        }
        sizes = pd.Series(lab).value_counts().sort_index()
        row["cluster_sizes"] = str(sizes.to_dict())
        row["min_cluster_size"] = int(sizes.min())
        row["max_cluster_pct"] = float(sizes.max() / len(lab))
        rows.append(row)
        if sizes.min() == 1:
            warn(f"[{tag}] k={k}: cluster with only 1 DAO")
        if sizes.max() / len(lab) > 0.9:
            warn(f"[{tag}] k={k}: one cluster has >90% of DAOs")

    met = pd.DataFrame(rows)
    met.to_csv(out_dir / f"kmeans_metrics_{tag}.csv", index=False)

    # plots
    ks = met["k"]
    for col, fn in [
        ("inertia", "elbow"),
        ("silhouette", "silhouette_vs_k"),
        ("calinski_harabasz", "ch_vs_k"),
        ("davies_bouldin", "db_vs_k"),
    ]:
        plt.figure(figsize=(6, 4))
        plt.plot(ks, met[col], "o-")
        plt.xlabel("k")
        plt.ylabel(col)
        plt.title(f"{fn} ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir / f"kmeans_{fn}_{tag}.png", dpi=160)
        plt.close()

    # pick k=3 for detailed outputs (user can change) — use best silhouette
    k_best = int(met.loc[met["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=k_best, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(X_scaled)
    sizes = pd.Series(labels).value_counts().sort_index()
    pd.DataFrame({"cluster": sizes.index, "n": sizes.values}).to_csv(
        out_dir / f"kmeans_cluster_sizes_k{k_best}_{tag}.csv", index=False
    )

    centers_scaled = pd.DataFrame(km.cluster_centers_, columns=[f"z_{c}" for c in feat_cols])
    centers_scaled.to_csv(out_dir / f"kmeans_centroids_scaled_k{k_best}_{tag}.csv", index=False)

    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    pd.DataFrame(centers_orig, columns=feat_cols).to_csv(
        out_dir / f"kmeans_centroids_original_k{k_best}_{tag}.csv", index=False
    )

    df_plot = pd.DataFrame(X_orig, columns=feat_cols)
    df_plot[ID_COLUMN] = df_ids[ID_COLUMN].values
    df_plot["cluster"] = labels

    assign = df_plot[[ID_COLUMN, "cluster"]].copy()
    assign.insert(1, "k", k_best)
    assign.to_csv(out_dir / "cluster_assignments.csv", index=False)

    summ_rows = []
    for cl in sorted(np.unique(labels)):
        sub = df_plot.loc[df_plot["cluster"] == cl, feat_cols]
        row: Dict = {"cluster": int(cl), "n": int(len(sub))}
        for c in feat_cols:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_std"] = float(sub[c].std(ddof=1)) if len(sub) > 1 else np.nan
        summ_rows.append(row)
    pd.DataFrame(summ_rows).to_csv(out_dir / "cluster_summary_statistics.csv", index=False)

    ncols = 3
    nrows = int(np.ceil(len(feat_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, c in enumerate(feat_cols):
        sns.boxplot(data=df_plot, x="cluster", y=c, ax=axes[i])
        axes[i].set_title(c, fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"K-means k={k_best} ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"kmeans_boxplots_k{k_best}_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close()

    # PCA colored by cluster
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", alpha=0.7, s=25)
    plt.colorbar(scatter, label="cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA by K-means cluster k={k_best} ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"kmeans_pca_scatter_k{k_best}_{tag}.png", dpi=180)
    plt.close()

    return met


# =============================================================================
# 8. Hierarchical
# =============================================================================


def run_hierarchical_validation(
    X_scaled: np.ndarray,
    feat_cols: List[str],
    kmeans_labels: np.ndarray,
    k_compare: int,
    tag: str,
    out_dir: Path,
) -> pd.DataFrame:
    Z = linkage(X_scaled, method="ward")
    plt.figure(figsize=(14, 5))
    dendrogram(Z, no_labels=True)
    plt.title(f"Ward dendrogram ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"hierarchical_dendrogram_{tag}.png", dpi=160)
    plt.close()

    rows = []
    for k in K_RANGE:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = hc.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X_scaled, lab),
                "calinski_harabasz": calinski_harabasz_score(X_scaled, lab),
                "davies_bouldin": davies_bouldin_score(X_scaled, lab),
                "sizes": str(pd.Series(lab).value_counts().to_dict()),
            }
        )
    hmet = pd.DataFrame(rows)
    hmet.to_csv(out_dir / f"hierarchical_metrics_{tag}.csv", index=False)

    hc3 = AgglomerativeClustering(n_clusters=k_compare, linkage="ward").fit_predict(X_scaled)
    ct = pd.crosstab(pd.Series(kmeans_labels, name="kmeans"), pd.Series(hc3, name="hierarchical"))
    ct.to_csv(out_dir / f"kmeans_vs_hierarchical_crosstab_{tag}.csv")
    return hmet


# =============================================================================
# 9. Density clustering
# =============================================================================


def run_density_clustering(X_scaled: np.ndarray, tag: str, out_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    db_rows = []
    for eps in DBSCAN_EPS_GRID:
        for ms in DBSCAN_MIN_SAMPLES_GRID:
            db = DBSCAN(eps=eps, min_samples=ms)
            lab = db.fit_predict(X_scaled)
            nclust = len(set(lab)) - (1 if -1 in lab else 0)
            noise = int((lab == -1).sum())
            db_rows.append(
                {
                    "eps": eps,
                    "min_samples": ms,
                    "n_clusters": nclust,
                    "n_noise": noise,
                    "labels_preview": str(np.unique(lab).tolist()),
                }
            )
    db_df = pd.DataFrame(db_rows)
    db_df.to_csv(out_dir / f"dbscan_results_{tag}.csv", index=False)

    hdb_df = None
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(X_scaled) // 100), min_samples=5)
        hl = clusterer.fit_predict(X_scaled)
        probs = getattr(clusterer, "probabilities_", None)
        hdb_df = pd.DataFrame(
            {
                "n_clusters": [len(set(hl)) - (1 if -1 in hl else 0)],
                "n_noise": [int((hl == -1).sum())],
                "has_probabilities": probs is not None,
            }
        )
        hdb_df.to_csv(out_dir / f"hdbscan_results_{tag}.csv", index=False)
        log("HDBSCAN available; saved hdbscan_results.")
    except ImportError:
        warn("hdbscan not installed; skipped HDBSCAN.")
    return db_df, hdb_df


# =============================================================================
# 10. Null benchmark
# =============================================================================


def run_null_benchmark(X_scaled: np.ndarray, tag: str, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    real_scores = []
    null_scores = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        rl = km.fit_predict(X_scaled)
        real_scores.append(
            {
                "k": k,
                "silhouette": silhouette_score(X_scaled, rl),
                "ch": calinski_harabasz_score(X_scaled, rl),
                "db": davies_bouldin_score(X_scaled, rl),
            }
        )

    for _ in range(NULL_N_PERM):
        Xn = X_scaled.copy()
        for j in range(Xn.shape[1]):
            Xn[:, j] = rng.permutation(Xn[:, j])
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
            nl = km.fit_predict(Xn)
            null_scores.append(
                {
                    "k": k,
                    "silhouette": silhouette_score(Xn, nl),
                    "ch": calinski_harabasz_score(Xn, nl),
                    "db": davies_bouldin_score(Xn, nl),
                }
            )

    real_df = pd.DataFrame(real_scores)
    null_df = pd.DataFrame(null_scores)
    summary = real_df.merge(
        null_df.groupby("k").agg({"silhouette": "mean", "ch": "mean", "db": "mean"}).reset_index(),
        on="k",
        suffixes=("_real", "_null_mean"),
    )
    summary.to_csv(out_dir / f"null_benchmark_{tag}.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(summary["k"], summary["silhouette_real"], "o-", label="real")
    plt.plot(summary["k"], summary["silhouette_null_mean"], "s--", label="null (permute cols)")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.legend()
    plt.title(f"Null vs real silhouette ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"null_vs_real_silhouette_{tag}.png", dpi=160)
    plt.close()
    return summary


# =============================================================================
# 11. Stability
# =============================================================================


def run_stability_analysis(X_scaled: np.ndarray, k: int, tag: str, out_dir: Path) -> pd.DataFrame:
    labels_list = []
    for seed in STABILITY_SEEDS:
        km = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT)
        labels_list.append(km.fit_predict(X_scaled))

    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    stab = pd.DataFrame({"pairwise_ari": aris})
    stab.to_csv(out_dir / f"stability_summary_{tag}.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(stab["pairwise_ari"], bins=15, edgecolor="black")
    plt.xlabel("Pairwise ARI across seeds")
    plt.title(f"K-means stability k={k} ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"stability_ari_hist_{tag}.png", dpi=160)
    plt.close()
    return stab


# =============================================================================
# 12. Reports
# =============================================================================


def _df_to_md_block(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string() + "\n```"


def write_summary_report(
    out_dir: Path,
    feat_summary: pd.DataFrame,
    n_all: int,
    n_no: int,
    n_out: int,
    high_corr: pd.DataFrame,
    km_all: pd.DataFrame,
    km_no: pd.DataFrame,
    null_all: pd.DataFrame,
    stab: pd.DataFrame,
) -> None:
    lines = [
        "# Clustering validation summary",
        "",
        f"- Sample size (cleaned, with outliers in data): **{n_all}**",
        f"- Sample size (outliers removed): **{n_no}**",
        f"- Rows flagged as outliers: **{n_out}**",
        "",
        "## Feature skewness",
        _df_to_md_block(feat_summary[["feature", "skewness", "skew_note"]]),
        "",
        "## Highly correlated pairs (>|0.8|)",
        _df_to_md_block(high_corr) if len(high_corr) else "(none)",
        "",
        "## K-means (all sample) — metrics by k",
        _df_to_md_block(km_all),
        "",
        "## K-means (no outliers) — metrics by k",
        _df_to_md_block(km_no),
        "",
        "## Null benchmark (silhouette real vs null mean)",
        _df_to_md_block(null_all),
        "",
        "## Stability (pairwise ARI)",
        _df_to_md_block(stab.describe()),
        "",
    ]
    (out_dir / "clustering_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    # Plain-text duplicate for editors without Markdown preview
    (out_dir / "clustering_validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    pd.DataFrame(
        {
            "metric": ["n_all", "n_no_outliers", "n_outliers"],
            "value": [n_all, n_no, n_out],
        }
    ).to_csv(out_dir / "sample_sizes.csv", index=False)


def sync_cluster_results_deliverables(src_run_dir: Path, dest_dir: Path, tag: str) -> None:
    """Copy Step 7 outputs into outputs/cluster_results/{tag}/ (heatmap, dists, assignments, PCA)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copies = [
        ("correlation_heatmap.png", "correlation_heatmap.png"),
        ("cluster_assignments.csv", "cluster_assignments.csv"),
        ("cluster_summary_statistics.csv", "cluster_summary_statistics.csv"),
    ]
    for src_name, dst_name in copies:
        p = src_run_dir / src_name
        if p.exists():
            shutil.copy2(p, dest_dir / dst_name)
    fd = src_run_dir / "feature_distributions"
    if fd.is_dir():
        dst_fd = dest_dir / "feature_distributions"
        if dst_fd.exists():
            shutil.rmtree(dst_fd)
        shutil.copytree(fd, dst_fd)
    # PCA scatter colored by K-means cluster
    for p in sorted(src_run_dir.glob("kmeans_pca_scatter_k*.png")):
        shutil.copy2(p, dest_dir / "pca_scatter_clusters.png")
        break
    for i in (1, 2, 3):
        ev = src_run_dir / f"pca_eigenvector_PC{i}_{tag}.png"
        if ev.exists():
            shutil.copy2(ev, dest_dir / f"pca_eigenvector_PC{i}.png")


def write_interpretation(
    out_dir: Path,
    feat_summary: pd.DataFrame,
    n_out: int,
    n_all: int,
    km_all: pd.DataFrame,
    null_all: pd.DataFrame,
    stab: pd.DataFrame,
) -> None:
    max_skew = feat_summary["skewness"].abs().max()
    sil_real = km_all["silhouette"].max()
    sil_null = null_all["silhouette_null_mean"].max() if "silhouette_null_mean" in null_all.columns else np.nan
    mean_ari = stab["pairwise_ari"].mean() if len(stab) else np.nan

    text = f"""# Interpretation (automated, cautious)

## Research question
Whether DAO-level governance features show **meaningful discrete clusters** or instead a **continuous spectrum** possibly **dominated by scale, participation variance, and extreme observations**.

## Evidence summary

1. **Distribution shape**: The maximum absolute skewness across features is approximately **{max_skew:.2f}**. 
   Large skew and heavy tails are consistent with **continuous, non-Gaussian** variation rather than well-separated spherical groups.

2. **Outliers**: **{n_out}** of **{n_all}** DAOs were flagged by the combined z-score / IQR rule. 
   If this count is large, **extreme DAOs may strongly influence** partition-based methods (e.g., K-means).

3. **K-means fit (max silhouette on full sample)**: **{sil_real:.3f}**. 
   Compared to **column-permutation null** benchmarks (mean silhouette **{sil_null:.3f}**), the real data {"appears somewhat stronger" if sil_real > sil_null + 0.02 else "is not dramatically stronger"} than random structure at the same dimensions.

4. **Stability**: Mean pairwise ARI across repeated K-means seeds is **{mean_ari:.3f}**. 
   Values well below 0.5 suggest **substantial sensitivity to initialization**, which weakens claims of a single stable typology.

## Conclusion (non-overclaiming)

Based on these diagnostics alone, one should **not** conclude that crisp, stable “DAO archetypes” exist unless **multiple methods agree**, **outlier sensitivity checks** support the same groups, and **substantive interpretation** aligns. 
If silhouette is modest, cluster sizes are imbalanced, and stability is low, the more defensible narrative is often a **high-dimensional continuum with outliers**, not discrete natural kinds.

*This text is a drafting aid; revise with domain knowledge and robustness checks reported elsewhere.*
"""
    (out_dir / "clustering_interpretation.md").write_text(text, encoding="utf-8")


# =============================================================================
# Main pipeline for one dataset version
# =============================================================================


def run_full_pipeline(
    df_ids: pd.DataFrame,
    df_feat: pd.DataFrame,
    feat_cols: List[str],
    tag: str,
    out_dir: Path,
    scaler_fit: Optional[RobustScaler] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = df_feat[feat_cols].to_numpy(float)
    scaler = scaler_fit or RobustScaler()
    Xs = scaler.fit_transform(X) if scaler_fit is None else scaler.transform(X)

    feat_sum = plot_feature_distributions(df_feat, feat_cols, out_dir)
    feat_sum.to_csv(out_dir / "feature_summary.csv", index=False)

    hp = compute_correlation_diagnostics(df_feat, feat_cols, out_dir)

    km = run_kmeans_validation(Xs, X, df_ids, feat_cols, scaler, tag, out_dir)
    k_best = int(km.loc[km["silhouette"].idxmax(), "k"])
    km_labels = KMeans(n_clusters=k_best, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(Xs)
    run_hierarchical_validation(Xs, feat_cols, km_labels, k_best, tag, out_dir)
    run_density_clustering(Xs, tag, out_dir)
    null_df = run_null_benchmark(Xs, tag, out_dir)
    stab_df = run_stability_analysis(Xs, k_best, tag, out_dir)
    run_pca_scaled(Xs, df_ids[ID_COLUMN], feat_cols, tag, out_dir)

    return feat_sum, hp, km, null_df, stab_df


def main() -> None:
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    raw = ensure_log_n_unique_voters(raw)
    log(f"Rows: {len(raw)}, columns: {len(raw.columns)}")
    print("Feature names (subset):", FEATURE_COLUMNS)

    df_ids, df_feat, scaler, feat_cols = preprocess_data(raw, FEATURE_COLUMNS, LOG1P_COLUMNS)
    print("\n=== Missing values (raw, selected columns) ===")
    print(raw[feat_cols].isna().sum())

    # Save cleaned
    cleaned = pd.concat([df_ids, df_feat], axis=1)
    cleaned.to_csv(out / "cleaned_dataset.csv", index=False)

    # Outliers on full feature frame
    report, df_all_flag, df_clean = detect_outliers(df_ids, df_feat, feat_cols, out)
    df_clean.to_csv(out / "cleaned_dataset_no_outliers.csv", index=False)

    feat_all, hp_all, km_all, null_all, stab_all = run_full_pipeline(
        df_ids, df_feat, feat_cols, "all", out / "all"
    )
    feat_no, hp_no, km_no, null_no, stab_no = run_full_pipeline(
        df_clean[[ID_COLUMN]].reset_index(drop=True),
        df_clean[feat_cols].reset_index(drop=True),
        feat_cols,
        "no_outliers",
        out / "no_outliers",
        scaler_fit=None,
    )

    # PCA on pre-built scaled for consistency (already inside run_full_pipeline)

    write_summary_report(
        out,
        feat_all,
        len(cleaned),
        len(df_clean),
        int(report["outlier_flag"].sum()),
        hp_all,
        km_all,
        km_no,
        null_all,
        stab_all,
    )
    write_interpretation(out, feat_all, int(report["outlier_flag"].sum()), len(cleaned), km_all, null_all, stab_all)

    CLUSTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sync_cluster_results_deliverables(out / "all", CLUSTER_RESULTS_DIR / "all", "all")
    sync_cluster_results_deliverables(out / "no_outliers", CLUSTER_RESULTS_DIR / "no_outliers", "no_outliers")
    log(f"Step 7 bundle: {CLUSTER_RESULTS_DIR}")

    # Copy key files to root names per demand §14
    shutil.copy(out / "all" / "kmeans_metrics_all.csv", out / "kmeans_metrics.csv")
    shutil.copy(out / "all" / "hierarchical_metrics_all.csv", out / "hierarchical_metrics.csv")
    shutil.copy(out / "all" / "dbscan_results_all.csv", out / "dbscan_results.csv")
    shutil.copy(out / "all" / "null_benchmark_all.csv", out / "null_benchmark.csv")
    shutil.copy(out / "all" / "stability_summary_all.csv", out / "stability_summary.csv")
    shutil.copy(out / "all" / "correlation_matrix.csv", out / "correlation_matrix.csv")
    shutil.copy(out / "all" / "high_corr_pairs.csv", out / "high_corr_pairs.csv")
    shutil.copy(out / "all" / "feature_summary.csv", out / "feature_summary.csv")
    shutil.copy(out / "all" / "pca_explained_variance_all.csv", out / "pca_explained_variance.csv")
    shutil.copy(out / "all" / "cluster_assignments.csv", out / "cluster_assignments.csv")
    shutil.copy(out / "all" / "cluster_summary_statistics.csv", out / "cluster_summary_statistics.csv")

    log(f"Done. Outputs: {out}")


if __name__ == "__main__":
    main()
