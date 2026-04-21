"""
Voter–space clustering: Mode A (per DAO cluster) or Mode B (global + dao_cluster feature).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from .config import (
    CLUSTERING_MODE,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    INCLUDE_DAO_CLUSTER_AS_FEATURE_MODE_B,
    KMEANS_K_MAX,
    KMEANS_K_MIN,
    N_INIT,
    OUTPUT_VOTER_DIR,
    RANDOM_STATE,
    log,
)

FEATURE_COLS: List[str] = [
    "log_total_votes",
    "avg_voting_power",
    "pct_for_votes",
    "pct_against_votes",
    "pct_abstain_votes",
    "pct_aligned_with_majority",
    "is_whale_ratio",
    "participation_rate",
    "active_span_days",
    "vote_frequency",
    "vote_entropy",
    "std_voting_power",
    "n_daos_participated",
]


def _fill_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median()
        out[c] = out[c].fillna(med)
    return out


def _kmeans_best_k(X: np.ndarray) -> Tuple[int, np.ndarray, float]:
    best_k, best_lab, best_sil = KMEANS_K_MIN, None, -1.0
    for k in range(KMEANS_K_MIN, KMEANS_K_MAX + 1):
        if X.shape[0] <= k + 1:
            break
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        lab = km.fit_predict(X)
        try:
            sil = float(silhouette_score(X, lab))
        except ValueError:
            sil = float("nan")
        if sil > best_sil:
            best_sil, best_k, best_lab = sil, k, lab
    if best_lab is None:
        km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=N_INIT)
        best_lab = km.fit_predict(X)
        best_k = 2
        best_sil = float(silhouette_score(X, best_lab)) if X.shape[0] > 2 else float("nan")
    return best_k, best_lab, best_sil


def _try_hdbscan(X: np.ndarray) -> Optional[Tuple[np.ndarray, int, int]]:
    try:
        import hdbscan

        cl = hdbscan.HDBSCAN(
            min_cluster_size=min(HDBSCAN_MIN_CLUSTER_SIZE, max(5, X.shape[0] // 20)),
            min_samples=min(HDBSCAN_MIN_SAMPLES, max(3, X.shape[0] // 50)),
        )
        lab = cl.fit_predict(X)
        ncl = len(set(lab)) - (1 if -1 in lab else 0)
        noise = int((lab == -1).sum())
        return lab, ncl, noise
    except ImportError:
        return None


def run_clustering_and_save(
    filtered_features_csv: Path,
    assignments_out_csv: Path,
    summary_out_csv: Path,
    figures_dir: Path,
) -> None:
    filt = pd.read_csv(filtered_features_csv)
    if filt.empty:
        raise ValueError("Filtered feature table is empty; lower MIN_VOTES_PER_VOTER_SPACE or increase data.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_VOTER_DIR.mkdir(parents=True, exist_ok=True)

    rows_assign: List[dict] = []
    rows_summary: List[dict] = []
    metrics_rows: List[dict] = []

    mode = CLUSTERING_MODE.upper().strip()
    log(f"Clustering mode: {mode}")

    if mode == "A":
        dao_vals = sorted(filt["dao_cluster"].dropna().unique().tolist())
        for dc in dao_vals:
            sub = filt[filt["dao_cluster"] == dc].copy().reset_index(drop=True)
            if len(sub) < max(30, (KMEANS_K_MAX + 1) * 5):
                log(f"Skip dao_cluster={dc}: n={len(sub)} too small.")
                continue
            cols_use = list(FEATURE_COLS)
            sub_f = _fill_numeric(sub, cols_use)
            X = sub_f[cols_use].to_numpy(float)
            scaler = RobustScaler()
            Xs = scaler.fit_transform(X)
            best_k, lab, sil = _kmeans_best_k(Xs)
            for i in range(len(sub_f)):
                rows_assign.append(
                    {
                        "voter": sub_f.loc[i, "voter"],
                        "space": sub_f.loc[i, "space"],
                        "dao_cluster": int(dc) if pd.notna(dc) else dc,
                        "voter_cluster": int(lab[i]),
                        "clustering_method": "kmeans",
                        "clustering_mode": "A",
                    }
                )
            metrics_rows.append(
                {"dao_cluster": dc, "method": "kmeans", "best_k": best_k, "silhouette": sil, "n_rows": len(sub)}
            )
            # summary for this dao_cluster × voter_cluster
            tmp = sub_f.copy()
            tmp["voter_cluster"] = lab
            for vc in sorted(set(lab.tolist())):
                part = tmp[tmp["voter_cluster"] == vc]
                row: Dict = {
                    "dao_cluster": dc,
                    "voter_cluster": int(vc),
                    "clustering_method": "kmeans",
                    "clustering_mode": "A",
                    "n_rows": len(part),
                }
                for c in cols_use:
                    row[f"mean_{c}"] = float(part[c].mean())
                rows_summary.append(row)

            hdb = _try_hdbscan(Xs)
            if hdb is not None:
                hlab, ncl, noise = hdb
                metrics_rows.append(
                    {
                        "dao_cluster": dc,
                        "method": "hdbscan",
                        "best_k": ncl,
                        "silhouette": float("nan"),
                        "n_rows": len(sub),
                        "noise_points": noise,
                    }
                )
                for i in range(len(sub_f)):
                    rows_assign.append(
                        {
                            "voter": sub_f.loc[i, "voter"],
                            "space": sub_f.loc[i, "space"],
                            "dao_cluster": int(dc) if pd.notna(dc) else dc,
                            "voter_cluster": int(hlab[i]),
                            "clustering_method": "hdbscan",
                            "clustering_mode": "A",
                        }
                    )

            _plot_pca_mode_a(sub_f, lab, int(dc), cols_use, figures_dir / f"pca_dao_cluster_{dc}_kmeans.png")

    elif mode == "B":
        cols_use = list(FEATURE_COLS)
        sub = filt.copy().reset_index(drop=True)
        if INCLUDE_DAO_CLUSTER_AS_FEATURE_MODE_B and "dao_cluster" in sub.columns:
            sub["_dao_cluster_num"] = pd.to_numeric(sub["dao_cluster"], errors="coerce").fillna(-1)
            cols_use = cols_use + ["_dao_cluster_num"]
        sub_f = _fill_numeric(sub, [c for c in cols_use if c in sub.columns])
        X = sub_f[cols_use].to_numpy(float)
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        best_k, lab, sil = _kmeans_best_k(Xs)
        log(f"Mode B KMeans best_k={best_k}, silhouette={sil:.4f}")
        for i in range(len(sub_f)):
            rows_assign.append(
                {
                    "voter": sub_f.loc[i, "voter"],
                    "space": sub_f.loc[i, "space"],
                    "dao_cluster": sub_f.loc[i, "dao_cluster"],
                    "voter_cluster": int(lab[i]),
                    "clustering_method": "kmeans",
                    "clustering_mode": "B",
                }
            )
        metrics_rows.append({"dao_cluster": "all", "method": "kmeans", "best_k": best_k, "silhouette": sil, "n_rows": len(sub)})
        tmp = sub_f.copy()
        tmp["voter_cluster"] = lab
        for vc in sorted(set(lab.tolist())):
            part = tmp[tmp["voter_cluster"] == vc]
            row = {
                "dao_cluster": "all",
                "voter_cluster": int(vc),
                "clustering_method": "kmeans",
                "clustering_mode": "B",
                "n_rows": len(part),
            }
            for c in FEATURE_COLS:
                if c in part.columns:
                    row[f"mean_{c}"] = float(part[c].mean())
            rows_summary.append(row)

        _plot_pca_mode_b(sub_f, lab, FEATURE_COLS, figures_dir / "pca_mode_b_kmeans.png")
        hdb = _try_hdbscan(Xs)
        if hdb is not None:
            hlab, ncl, noise = hdb
            log(f"Mode B HDBSCAN clusters={ncl}, noise={noise}")
            for i in range(len(sub_f)):
                rows_assign.append(
                    {
                        "voter": sub_f.loc[i, "voter"],
                        "space": sub_f.loc[i, "space"],
                        "dao_cluster": sub_f.loc[i, "dao_cluster"],
                        "voter_cluster": int(hlab[i]),
                        "clustering_method": "hdbscan",
                        "clustering_mode": "B",
                    }
                )
    else:
        raise ValueError(f"Unknown CLUSTERING_MODE={CLUSTERING_MODE}; use A or B.")

    assign_df = pd.DataFrame(rows_assign)
    assign_df.to_csv(assignments_out_csv, index=False)
    log(f"Saved assignments: {assignments_out_csv} (rows={len(assign_df)})")

    summ_df = pd.DataFrame(rows_summary)
    summ_df.to_csv(summary_out_csv, index=False)
    log(f"Saved cluster summary: {summary_out_csv}")

    pd.DataFrame(metrics_rows).to_csv(OUTPUT_VOTER_DIR / "kmeans_sweep_metrics.csv", index=False)

    _plot_feature_dists(assign_df, filt, figures_dir)


def _plot_pca_mode_a(sub_f: pd.DataFrame, labels: np.ndarray, dc: int, cols: List[str], out_path: Path) -> None:
    X = sub_f[cols].to_numpy(float)
    if X.shape[0] < 5:
        return
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = pca.fit_transform(RobustScaler().fit_transform(X))
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", alpha=0.65, s=18)
    plt.colorbar(sc, label="voter_cluster")
    plt.title(f"PCA (voter–space) | dao_cluster={dc} | KMeans")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pca_mode_b(sub_f: pd.DataFrame, labels: np.ndarray, cols: List[str], out_path: Path) -> None:
    use = [c for c in cols if c in sub_f.columns]
    X = sub_f[use].to_numpy(float)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = pca.fit_transform(RobustScaler().fit_transform(X))
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", alpha=0.65, s=18)
    plt.colorbar(sc, label="voter_cluster")
    plt.title("PCA (voter–space) | Mode B | KMeans")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_feature_dists(assign_df: pd.DataFrame, filt: pd.DataFrame, figures_dir: Path) -> None:
    """Histograms of key features by voter_cluster (KMeans rows only)."""
    km = assign_df[assign_df["clustering_method"] == "kmeans"].copy()
    if km.empty:
        return
    merged = km.merge(filt, on=["voter", "space"], how="left", suffixes=("", "_feat"))
    key_feats = ["log_total_votes", "avg_voting_power", "pct_for_votes", "vote_entropy"]
    for col in key_feats:
        if col not in merged.columns:
            continue
        plt.figure(figsize=(8, 5))
        for cl in sorted(merged["voter_cluster"].unique()):
            sub = merged[merged["voter_cluster"] == cl][col].dropna()
            if len(sub) == 0:
                continue
            sns.kdeplot(sub, label=f"C{cl}")
        plt.legend(title="voter_cluster")
        plt.title(f"Distribution of {col} by voter_cluster (KMeans)")
        plt.tight_layout()
        plt.savefig(figures_dir / f"dist_{col}_by_voter_cluster.png", dpi=140)
        plt.close()

