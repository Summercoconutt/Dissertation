# Behaviour Modelling Pipeline

This folder contains an end-to-end pipeline from existing clustering outputs to behaviour modelling training and evaluation.

## Inputs expected from `Voter_Clustering`

- `Voter_Clustering/data/processed/master_votes_with_dao_cluster.parquet`
- `Voter_Clustering/outputs/voter_clustering/voter_cluster_assignments.csv`

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run full behaviour modelling pipeline:

```bash
python run_pipeline.py
```

This will:
- build a behaviour dataset with both DAO cluster and voter cluster labels
- train a temporal classifier
- evaluate on validation split
- save artifacts and metrics

## Output folders

- `outputs/behaviour_modelling/behaviour_dataset.csv`
- `outputs/behaviour_modelling/agent2_artifacts/`
- `outputs/behaviour_modelling/eval/`

## Server notes

You can override default paths via environment variables:

- `BM_MASTER_VOTES_PARQUET`
- `BM_VOTER_CLUSTER_ASSIGNMENTS_CSV`
- `BM_OUTPUT_DIR`
