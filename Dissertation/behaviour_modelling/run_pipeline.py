#!/usr/bin/env python3
"""
Run full behaviour modelling pipeline:
1) Build dataset from clustering outputs
2) Train model
3) Evaluate model
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from build_dataset import build_behaviour_dataset


def main() -> None:
    root = Path(__file__).resolve().parent
    default_master = root.parent / "Voter_Clustering" / "data" / "processed" / "master_votes_with_dao_cluster.parquet"
    default_assign = root.parent / "Voter_Clustering" / "outputs" / "voter_clustering" / "voter_cluster_assignments.csv"
    default_out = root / "outputs" / "behaviour_modelling"

    master_path = Path(os.environ.get("BM_MASTER_VOTES_PARQUET", str(default_master)))
    assign_path = Path(os.environ.get("BM_VOTER_CLUSTER_ASSIGNMENTS_CSV", str(default_assign)))
    output_dir = Path(os.environ.get("BM_OUTPUT_DIR", str(default_out)))
    dataset_csv = output_dir / "behaviour_dataset.csv"

    print("[1/3] Building behaviour dataset...")
    build_behaviour_dataset(
        master_votes_with_cluster_parquet=master_path,
        voter_cluster_assignments_csv=assign_path,
        output_csv=dataset_csv,
    )

    print("[2/3] Training model...")
    subprocess.run(
        [
            sys.executable,
            str(root / "train.py"),
            "--dataset_csv",
            str(dataset_csv),
            "--output_dir",
            str(output_dir / "agent2_artifacts"),
        ],
        check=True,
    )

    print("[3/3] Evaluating model...")
    subprocess.run(
        [
            sys.executable,
            str(root / "evaluate.py"),
            "--dataset_csv",
            str(dataset_csv),
            "--artifacts_dir",
            str(output_dir / "agent2_artifacts"),
            "--output_dir",
            str(output_dir / "eval"),
        ],
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
