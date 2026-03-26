from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from utils import log


def evaluate_clustering(x: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    # Need at least 2 clusters and at least one sample per cluster.
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan, np.nan
    if x.shape[0] <= len(uniq):
        return np.nan, np.nan
    sil = silhouette_score(x, labels)
    dbi = davies_bouldin_score(x, labels)
    return float(sil), float(dbi)


def safe_k_values(n_samples: int, k_min: int, k_max: int) -> list[int]:
    high = min(k_max, max(2, n_samples - 1))
    if high < k_min:
        return []
    return list(range(k_min, high + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=r"data\processed\dao_feature_matrix.parquet")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--run-gmm", action="store_true")
    parser.add_argument("--out-dir", default=r"outputs\cluster_results")
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.in_path))
    if "space" not in df.columns:
        raise ValueError("Input feature matrix must contain `space` column.")

    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError("No numeric features available for clustering.")
    x = numeric.fillna(numeric.median(numeric_only=True)).to_numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    assignment_frames = []
    k_values = safe_k_values(x.shape[0], args.k_min, args.k_max)
    if not k_values:
        warn_msg = f"Not enough DAO rows ({x.shape[0]}) for clustering with k>=2. Writing diagnostics only."
        log(warn_msg)
        pd.DataFrame({"message": [warn_msg]}).to_csv(out_dir / "clustering_note.csv", index=False)
        return

    # KMeans + Hierarchical (+ optional GMM)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=30)
        km_labels = km.fit_predict(x)
        km_sil, km_dbi = evaluate_clustering(x, km_labels)
        eval_rows.append({"method": "kmeans", "k": k, "silhouette": km_sil, "davies_bouldin": km_dbi})
        assignment_frames.append(
            pd.DataFrame({"space": df["space"], "method": "kmeans", "k": k, "cluster": km_labels})
        )

        hc = AgglomerativeClustering(n_clusters=k)
        hc_labels = hc.fit_predict(x)
        hc_sil, hc_dbi = evaluate_clustering(x, hc_labels)
        eval_rows.append({"method": "hierarchical", "k": k, "silhouette": hc_sil, "davies_bouldin": hc_dbi})
        assignment_frames.append(
            pd.DataFrame({"space": df["space"], "method": "hierarchical", "k": k, "cluster": hc_labels})
        )

        if args.run_gmm:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm_labels = gmm.fit_predict(x)
            gmm_sil, gmm_dbi = evaluate_clustering(x, gmm_labels)
            eval_rows.append({"method": "gmm", "k": k, "silhouette": gmm_sil, "davies_bouldin": gmm_dbi})
            assignment_frames.append(
                pd.DataFrame({"space": df["space"], "method": "gmm", "k": k, "cluster": gmm_labels})
            )

    eval_df = pd.DataFrame(eval_rows).sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    assignments = pd.concat(assignment_frames, ignore_index=True)

    eval_df.to_csv(out_dir / "clustering_evaluation.csv", index=False)
    assignments.to_csv(out_dir / "cluster_assignments_all_methods.csv", index=False)

    best = eval_df.dropna(subset=["silhouette"]).head(1)
    if not best.empty:
        best_method = best.iloc[0]["method"]
        best_k = int(best.iloc[0]["k"])
        best_assign = assignments[(assignments["method"] == best_method) & (assignments["k"] == best_k)].copy()
        best_assign.to_csv(out_dir / "cluster_assignments_best.csv", index=False)

        # Cluster summary statistics
        f = df.merge(best_assign[["space", "cluster"]], on="space", how="left")
        summary = f.groupby("cluster", as_index=False).agg(
            n_daos=("space", "nunique"),
            **{col: (col, "mean") for col in numeric.columns},
        )
        summary.to_csv(out_dir / "cluster_summary_stats.csv", index=False)

        # PCA visualization
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(x)
        plot_df = pd.DataFrame({"pc1": emb[:, 0], "pc2": emb[:, 1], "space": df["space"]})
        plot_df = plot_df.merge(best_assign[["space", "cluster"]], on="space", how="left")

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(plot_df["pc1"], plot_df["pc2"], c=plot_df["cluster"], cmap="tab10")
        for _, r in plot_df.iterrows():
            plt.text(r["pc1"], r["pc2"], str(r["space"]), fontsize=7, alpha=0.7)
        plt.title(f"Best clustering via PCA ({best_method}, k={best_k})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(out_dir / "clusters_pca.png", dpi=180)
        plt.close()

    log(f"Saved clustering evaluation: {out_dir / 'clustering_evaluation.csv'}")
    log(f"Saved all assignments: {out_dir / 'cluster_assignments_all_methods.csv'}")


if __name__ == "__main__":
    main()

