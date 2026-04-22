# Test pipeline (demo)

Small end-to-end run on a **subset of DAOs** to verify code before full server runs.

## Prerequisites

1. Complete or partial master table with DAO labels:
   - `../Voter_Clustering/data/processed/master_votes_with_dao_cluster.parquet`

2. DAO clustering assignments CSV (path from `Voter_Clustering` config or env `VOTER_DAO_CLUSTERS_CSV`).

3. Dependencies (same stack as clustering + behaviour modelling):

```bash
pip install -r ../Voter_Clustering/requirements.txt
pip install -r ../behaviour_modelling/requirements.txt
```

## Run

From this folder:

```bash
cd test_pipeline
python run_demo.py
```

- Without `demo_spaces.txt`, the script picks the **top 5 spaces by vote count** from the master parquet.
- To fix which DAOs to use, copy `demo_spaces.example.txt` to `demo_spaces.txt` and list one `space` value per line (must match the `space` column in your master parquet).

### Faster / smaller GPU demo

```bash
python run_demo.py --epochs 1 --batch-size 8 --window 3 --max-length 64 --pretrained distilroberta-base --auto-spaces 3
```

### If Mode A skips all DAO slices

Increase data (more spaces / full master) or lower the demo threshold:

```bash
set VOTER_CLUSTER_MIN_ROWS_MODE_A=10
python run_demo.py
```

(Linux/macOS: `export VOTER_CLUSTER_MIN_ROWS_MODE_A=10`)

## Outputs

All under `test_pipeline/outputs/`:

| Path | Content |
|------|---------|
| `processed/demo_master_filtered.parquet` | Votes restricted to chosen spaces |
| `voter_clustering/` | Features, assignments, figures |
| `behaviour_modelling/behaviour_dataset.csv` | Vote rows + clusters for modelling |
| `behaviour_modelling/agent2_artifacts/` | Trained checkpoint + tokenizer |
| `behaviour_modelling/eval/` | Metrics CSVs |
| `demo_summary.json` | Paths and spaces used |
