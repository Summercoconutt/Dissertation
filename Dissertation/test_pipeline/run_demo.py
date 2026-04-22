#!/usr/bin/env python3
"""
Smoke-test a small subset of DAOs: voter clustering + behaviour modelling.

Run from Dissertation/test_pipeline:

  cd path/to/Dissertation/test_pipeline
  python run_demo.py

Requirements:
  - Master votes: parquet or CSV with columns expected by voter clustering
    (e.g. ../Voter_Clustering/data/processed/master_votes_sample.csv or full master parquet).
  - DAO cluster_assignments.csv so spaces get dao_cluster (see --dao-assignments).

Environment (optional):
  VOTER_CLUSTER_MIN_ROWS_MODE_A  Lower threshold for Mode-A slice size (demo default: 15)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DISSERTATION_ROOT = SCRIPT_DIR.parent
VC_ROOT = DISSERTATION_ROOT / "Voter_Clustering"
BM_ROOT = DISSERTATION_ROOT / "behaviour_modelling"


def parse_demo_spaces(spaces_file: Path | None) -> list[str] | None:
    """Return list from demo_spaces.txt, or None if file missing / empty."""
    path = spaces_file or (SCRIPT_DIR / "demo_spaces.txt")
    if not path.exists():
        return None
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines if lines else None


def pick_top_spaces_by_volume(master_path: Path, n: int, dao_assignments_csv: Path | None) -> list[str]:
    suf = master_path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(master_path, usecols=["space"])
    else:
        df = pd.read_parquet(master_path, columns=["space"])
    counts = df["space"].astype(str).value_counts()
    if dao_assignments_csv is not None and dao_assignments_csv.exists():
        dao_spaces = set(pd.read_csv(dao_assignments_csv)["space"].astype(str))
        counts = counts[counts.index.isin(dao_spaces)]
    if counts.empty:
        raise ValueError(
            "No overlapping spaces between master and DAO cluster_assignments.csv. "
            "Use demo_spaces.txt with space names present in both, or fix --master / --dao-assignments."
        )
    return counts.head(n).index.astype(str).tolist()


def filter_master_by_spaces(master_path: Path, spaces: list[str], out_parquet: Path) -> tuple[Path, int]:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    spaces_set = set(spaces)
    suf = master_path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(master_path)
    else:
        df = pd.read_parquet(master_path)
    if "space" not in df.columns:
        raise ValueError("Master must contain column 'space'.")
    filtered = df[df["space"].isin(spaces_set)].copy()
    if filtered.empty:
        raise ValueError(f"No rows after filtering by spaces={spaces}. Check names match snapshot folder names.")
    filtered.to_parquet(out_parquet, index=False)
    print(f"[demo] Filtered master: {len(filtered)} rows -> {out_parquet}")
    return out_parquet, len(filtered)


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo voter cluster + behaviour model on selected DAOs.")
    ap.add_argument(
        "--master",
        type=str,
        default=str(VC_ROOT / "data" / "processed" / "master_votes_sample.csv"),
        help="Master votes parquet or CSV (columns as in Voter_Clustering master).",
    )
    ap.add_argument(
        "--dao-assignments",
        type=str,
        default="",
        help="DAO cluster_assignments.csv (default: VOTER_DAO_CLUSTERS_CSV or Voter_Clustering config path).",
    )
    ap.add_argument("--spaces-file", type=str, default="", help="Optional list of spaces (default: demo_spaces.txt).")
    ap.add_argument("--auto-spaces", type=int, default=5, help="If no spaces file: pick top-N spaces by vote count.")
    ap.add_argument("--demo-min-votes-per-space", type=int, default=2, help="Feature filter min votes (voter,space).")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window", type=int, default=3, help="Behaviour window length (lower for tiny demos).")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--pretrained", type=str, default="distilroberta-base", help="Smaller/faster default for demo.")
    ap.add_argument(
        "--max-train-windows",
        type=int,
        default=512,
        help="Forwarded to behaviour train.py as --max_train_windows (smoke test cap).",
    )
    ap.add_argument(
        "--max-valid-windows",
        type=int,
        default=256,
        help="Forwarded to behaviour train.py as --max_valid_windows.",
    )
    ap.add_argument(
        "--eval-max-windows",
        type=int,
        default=512,
        help="Forwarded to evaluate.py --max_windows (cap eval inference).",
    )
    args = ap.parse_args()

    master_path = Path(args.master).resolve()
    if not master_path.exists():
        raise FileNotFoundError(
            f"Missing master file: {master_path}\nBuild master under Voter_Clustering or pass --master "
            f"(sample: {VC_ROOT / 'data' / 'processed' / 'master_votes_sample.csv'})."
        )

    dao_csv: Path | None
    if args.dao_assignments and str(args.dao_assignments).strip():
        dao_csv = Path(args.dao_assignments).resolve()
    else:
        env_dao = (os.environ.get("VOTER_DAO_CLUSTERS_CSV") or "").strip()
        if env_dao:
            dao_csv = Path(env_dao).expanduser().resolve()
        else:
            dao_csv = None
    if dao_csv is None:
        dao_csv = (
            DISSERTATION_ROOT
            / "DAO_Clustering_cleaning"
            / "outputs"
            / "cluster_results"
            / "no_outliers"
            / "cluster_assignments.csv"
        )
    if not dao_csv.exists():
        raise FileNotFoundError(
            f"DAO cluster assignments not found: {dao_csv}\nSet --dao-assignments or VOTER_DAO_CLUSTERS_CSV."
        )

    demo_out = SCRIPT_DIR / "outputs"
    proc_dir = demo_out / "processed"
    vc_out = demo_out / "voter_clustering"
    bm_out = demo_out / "behaviour_modelling"
    artifacts = bm_out / "agent2_artifacts"
    eval_out = bm_out / "eval"
    for d in (proc_dir, vc_out, bm_out):
        d.mkdir(parents=True, exist_ok=True)

    spaces_file = Path(args.spaces_file) if args.spaces_file else None
    manual_spaces = parse_demo_spaces(spaces_file)
    if manual_spaces:
        spaces = manual_spaces
        print(f"[demo] Using {len(spaces)} spaces from demo_spaces.txt")
    else:
        spaces = pick_top_spaces_by_volume(master_path, args.auto_spaces, dao_csv)
        print(f"[demo] Auto-selected top {len(spaces)} spaces by vote volume (also in DAO labels): {spaces}")

    dao_space_set = set(pd.read_csv(dao_csv)["space"].astype(str))
    unknown = [s for s in spaces if str(s) not in dao_space_set]
    if unknown:
        raise ValueError(
            f"These spaces are not in DAO cluster_assignments.csv: {unknown}. "
            "Fix demo_spaces.txt or --dao-assignments."
        )

    demo_master = proc_dir / "demo_master_filtered.parquet"
    _, n_master_rows = filter_master_by_spaces(master_path, spaces, demo_master)

    # Relax Mode-A minimum slice size for small demos (align with env if user set it).
    os.environ.setdefault("VOTER_CLUSTER_MIN_ROWS_MODE_A", "15")

    sys.path.insert(0, str(VC_ROOT / "src"))
    from voter_clustering.build_voter_space_features import build_voter_space_features
    from voter_clustering.run_voter_clustering import run_clustering_and_save

    merged_out = proc_dir / "demo_master_with_dao.parquet"
    feat_csv = vc_out / "voter_space_features.csv"
    feat_filt = vc_out / "voter_space_features_filtered.csv"

    print("[demo] Building (voter, space) features + clustering...")
    build_voter_space_features(
        master_parquet=demo_master,
        assignments_csv=dao_csv,
        out_with_cluster_parquet=merged_out,
        out_features_csv=feat_csv,
        out_features_filtered_csv=feat_filt,
        min_votes=args.demo_min_votes_per_space,
    )
    run_clustering_and_save(
        filtered_features_csv=feat_filt,
        assignments_out_csv=vc_out / "voter_cluster_assignments.csv",
        summary_out_csv=vc_out / "cluster_summary_statistics.csv",
        figures_dir=vc_out / "figures",
    )

    behaviour_csv = bm_out / "behaviour_dataset.csv"
    sys.path.insert(0, str(BM_ROOT))
    from build_dataset import build_behaviour_dataset

    print("[demo] Building behaviour dataset CSV...")
    build_behaviour_dataset(
        master_votes_with_cluster_parquet=merged_out,
        voter_cluster_assignments_csv=vc_out / "voter_cluster_assignments.csv",
        output_csv=behaviour_csv,
    )

    train_py = BM_ROOT / "train.py"
    eval_py = BM_ROOT / "evaluate.py"
    summary_path = demo_out / "demo_summary.json"

    print("[demo] Training behaviour model (short run)...")
    subprocess.run(
        [
            sys.executable,
            str(train_py),
            "--dataset_csv",
            str(behaviour_csv),
            "--output_dir",
            str(artifacts),
            "--pretrained",
            args.pretrained,
            "--window",
            str(args.window),
            "--max_length",
            str(args.max_length),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--max_train_windows",
            str(args.max_train_windows),
            "--max_valid_windows",
            str(args.max_valid_windows),
        ],
        cwd=str(BM_ROOT),
        check=True,
    )

    print("[demo] Evaluating...")
    subprocess.run(
        [
            sys.executable,
            str(eval_py),
            "--dataset_csv",
            str(behaviour_csv),
            "--artifacts_dir",
            str(artifacts),
            "--output_dir",
            str(eval_out),
            "--batch_size",
            str(args.batch_size),
            "--max_windows",
            str(args.eval_max_windows),
        ],
        cwd=str(BM_ROOT),
        check=True,
    )

    # Short summary for the supervisor bundle
    cfg_path = artifacts / "config.json"
    summary = {
        "spaces_used": spaces,
        "master_rows_filtered": n_master_rows,
        "behaviour_dataset_csv": str(behaviour_csv),
        "artifacts_dir": str(artifacts),
        "eval_dir": str(eval_out),
    }
    if cfg_path.exists():
        summary["train_config"] = json.loads(cfg_path.read_text(encoding="utf-8"))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("Demo finished OK.")
    print(f"  Voter cluster outputs: {vc_out}")
    print(f"  Behaviour dataset:     {behaviour_csv}")
    print(f"  Model + config:       {artifacts}")
    print(f"  Eval CSVs/plots base: {eval_out}")
    print(f"  Summary JSON:         {summary_path}")


if __name__ == "__main__":
    main()
