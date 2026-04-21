"""Configuration: edit paths or set env VOTER_SNAPSHOT_ROOT / VOTER_DAO_CLUSTERS_CSV."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


def _path_env(name: str, default: Path) -> Path:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else default


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw: .../spaces/space=*/proposal=*.parquet
SNAPSHOT_SPACES_ROOT = _path_env(
    "VOTER_SNAPSHOT_ROOT",
    Path(r"/path/to/snapshot_votes_data/snapshot_votes_441/spaces"),
)

# DAO-level cluster_assignments.csv (columns: space + cluster or dao_cluster)
DAO_CLUSTER_ASSIGNMENTS_CSV = _path_env(
    "VOTER_DAO_CLUSTERS_CSV",
    PROJECT_ROOT.parent
    / "DAO_Clustering_cleaning"
    / "outputs"
    / "cluster_results"
    / "no_outliers"
    / "cluster_assignments.csv",
)

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_VOTER_DIR = PROJECT_ROOT / "outputs" / "voter_clustering"

# None = all proposal parquet files; integer = smoke-test cap
MAX_PARQUET_FILES: Optional[int] = None
PARQUET_READ_BATCH = 80
MASTER_CSV_SAMPLE_ROWS = 50_000

RAW_TO_CANONICAL: Dict[str, str] = {
    "Space": "space",
    "Proposal ID": "proposal_id",
    "Proposal Title": "proposal_title",
    "Proposal Body": "proposal_body",
    "Created Time": "proposal_created",
    "Voter": "voter",
    "Choice": "choice_raw",
    "Voting Power": "voting_power",
    "VP Ratio (%)": "vp_ratio_pct",
    "Is Whale": "is_whale",
    "Aligned With Majority": "aligned_with_majority",
    "Vote Timestamp": "vote_timestamp",
    "FollowersCount": "followers_count",
}

CANONICAL_MASTER_COLUMNS: List[str] = [
    "space",
    "proposal_id",
    "voter",
    "choice_raw",
    "choice_norm",
    "voting_power",
    "vote_timestamp",
    "proposal_created",
    "proposal_title",
    "proposal_body",
    "followers_count",
    "vp_ratio_pct",
    "is_whale",
    "aligned_with_majority",
]

CHOICE_INT_TO_NORM: Dict[int, str] = {1: "for", 2: "against", 3: "abstain"}

MIN_VOTES_PER_VOTER_SPACE = 5

# A = cluster separately within each DAO cluster; B = global + dao_cluster feature
CLUSTERING_MODE = "A"
KMEANS_K_MIN = 2
KMEANS_K_MAX = 6
RANDOM_STATE = 42
N_INIT = 15
INCLUDE_DAO_CLUSTER_AS_FEATURE_MODE_B = True

# Optional
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 5

VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[voter_clustering] {msg}", flush=True)
