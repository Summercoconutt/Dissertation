"""
End-to-end runner: master votes -> standardize (inline) -> merge DAO clusters ->
(voter, space) features -> filter -> cluster -> figures.

Usage (from project root Voter_Clustering):
  python -m src.voter_clustering.main_voter_clustering

Or:
  cd src && python -m voter_clustering.main_voter_clustering
"""

from __future__ import annotations

from pathlib import Path

from .build_master_votes import build_master_votes
from .build_voter_space_features import build_voter_space_features
from .config import (
    DAO_CLUSTER_ASSIGNMENTS_CSV,
    DATA_PROCESSED_DIR,
    OUTPUT_VOTER_DIR,
    PROJECT_ROOT,
    log,
)
from .run_voter_clustering import run_clustering_and_save


def main() -> None:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VOTER_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_VOTER_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log("=== STEP 1: Build master vote-level table ===")
    master_pq = build_master_votes()

    log("=== STEP 3–4: Merge DAO clusters + build (voter, space) features ===")
    master_with_dc = DATA_PROCESSED_DIR / "master_votes_with_dao_cluster.parquet"
    feat_csv = OUTPUT_VOTER_DIR / "voter_space_features.csv"
    feat_filt_csv = OUTPUT_VOTER_DIR / "voter_space_features_filtered.csv"
    build_voter_space_features(
        master_parquet=master_pq,
        assignments_csv=DAO_CLUSTER_ASSIGNMENTS_CSV,
        out_with_cluster_parquet=master_with_dc,
        out_features_csv=feat_csv,
        out_features_filtered_csv=feat_filt_csv,
    )

    log("=== STEP 6–8: Clustering + figures ===")
    run_clustering_and_save(
        filtered_features_csv=feat_filt_csv,
        assignments_out_csv=OUTPUT_VOTER_DIR / "voter_cluster_assignments.csv",
        summary_out_csv=OUTPUT_VOTER_DIR / "cluster_summary_statistics.csv",
        figures_dir=figures_dir,
    )

    readme = OUTPUT_VOTER_DIR / "OUTPUT_FILES.txt"
    readme.write_text(
        "\n".join(
            [
                "Voter clustering outputs (paths relative to project root)",
                "",
                "data/processed/master_votes.parquet — unified vote rows",
                "data/processed/master_votes_sample.csv — optional head sample",
                "data/processed/master_votes_with_dao_cluster.parquet — votes + dao_cluster",
                "outputs/voter_clustering/voter_space_features.csv — full (voter, space) features",
                "outputs/voter_clustering/voter_space_features_filtered.csv — after min_votes filter",
                "outputs/voter_clustering/voter_cluster_assignments.csv — (voter, space) -> voter_cluster",
                "outputs/voter_clustering/cluster_summary_statistics.csv — cluster profiles",
                "outputs/voter_clustering/kmeans_sweep_metrics.csv — per-run k / silhouette notes",
                "outputs/voter_clustering/figures/ — PCA and distribution plots",
                "",
                "Edit src/voter_clustering/config.py: MAX_PARQUET_FILES=None for full scan (~57k files).",
            ]
        ),
        encoding="utf-8",
    )
    log(f"Done. See {readme}")


if __name__ == "__main__":
    main()
