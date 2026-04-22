# Thesis Pipeline Runbook (for supervisors / server environments)

This guide explains how to fetch the code on a server, configure data paths, and run the pipelines in order: **DAO clustering inputs → voter clustering → behaviour modelling**. Commands below assume **Linux** or **macOS** terminals; **Windows PowerShell** examples appear at the end.

---

## 1. Obtaining the code

Repository (confirm the branch name with your student if needed):

- **GitHub**: `https://github.com/Summercoconutt/Dissertation`
- **Recommended branch**: `thesis-pipeline` (includes the full `Dissertation/` tree in the repo; large binaries and generated artefacts are excluded from Git and must be produced on the server.)

Clone and enter the thesis folder:

```bash
git clone https://github.com/Summercoconutt/Dissertation.git
cd Dissertation
git checkout thesis-pipeline
```

Throughout this document, **`Dissertation/`** denotes that directory. Important subfolders:

| Path | Role |
|------|------|
| `DAO_Clustering_cleaning/outputs/.../cluster_assignments.csv` | DAO-level cluster labels (space → DAO cluster); used downstream by voter clustering and behaviour modelling |
| `Voter_Clustering/` | Build master vote tables from Snapshot data; voter clustering on top of DAO clusters |
| `behaviour_modelling/` | Behaviour modelling from clustered labels (train / evaluate) |
| `test_pipeline/` | **Small smoke test**: few DAOs, fast sanity check |

---

## 2. Environment setup

Use **Python 3.10+**. If CUDA is available, install a GPU build of PyTorch for faster behaviour modelling.

### 2.1 Virtual environment (recommended)

```bash
cd Dissertation
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2.2 Dependencies

Install both stacks:

```bash
pip install -U pip
pip install -r Voter_Clustering/requirements.txt
pip install -r behaviour_modelling/requirements.txt
```

Optional (if HDBSCAN is used in voter clustering):

```bash
pip install hdbscan
```

---

## 3. Data prerequisites

Two kinds of external inputs are required:

1. **Snapshot vote dump**  
   Folder layout must match what `Voter_Clustering` expects (root containing `space=*/proposal=*.parquet`; see notes inside `Voter_Clustering`). Your student should tell you the **absolute path** on the server.

2. **DAO cluster assignments CSV**  
   Must include **`space`** and a DAO cluster column named **`cluster`** or **`dao_cluster`**. Example path if unchanged in the repo:

   `Dissertation/DAO_Clustering_cleaning/outputs/cluster_results/no_outliers/cluster_assignments.csv`

If this CSV is not shipped (size or privacy), obtain the path from your student separately.

---

## 4. Recommended execution order

### Phase A — Smoke test (strongly recommended first)

**Goal**: Use **few DAOs** and **short training** to verify dependencies and paths without scanning the full Snapshot.

The repo ships **`Voter_Clustering/data/processed/master_votes_sample.csv`** (small vote sample). `run_demo.py` prefers spaces that also appear in `cluster_assignments.csv`, so DAO labels merge correctly.

```bash
cd Dissertation/test_pipeline
python run_demo.py --auto-spaces 2 --epochs 1 --batch-size 16 \
  --window 3 --max-length 48 --max-train-windows 256 --max-valid-windows 128 \
  --eval-max-windows 256
```

Outputs land under **`test_pipeline/outputs/`** (`voter_clustering/`, `behaviour_modelling/`, etc.). If Mode A skips clustering because slices are too small, lower the threshold:

```bash
export VOTER_CLUSTER_MIN_ROWS_MODE_A=10   # tune as needed
python run_demo.py ...
```

To pin DAOs manually: copy `demo_spaces.example.txt` to **`demo_spaces.txt`** and list one `space` per line (must match the `space` column in the master table).

---

### Phase B — Full voter clustering (build master from Snapshot)

After setting the Snapshot root and DAO CSV:

**Linux / macOS:**

```bash
export VOTER_SNAPSHOT_ROOT=/path/to/snapshot_votes_data/snapshot_votes_441/spaces
export VOTER_DAO_CLUSTERS_CSV=/path/to/cluster_assignments.csv
cd Dissertation/Voter_Clustering
pip install -r requirements.txt   # if not already
python run_pipeline.py
```

Main outputs (paths relative to `Voter_Clustering/`):

- `data/processed/master_votes.parquet`, `master_votes_with_dao_cluster.parquet`
- `outputs/voter_clustering/voter_cluster_assignments.csv`
- `outputs/voter_clustering/cluster_summary_statistics.csv`, `figures/`, etc.

Before a full scan, set **`MAX_PARQUET_FILES`** to an integer (e.g. `2000`) in **`src/voter_clustering/config.py`** for a trial run; use **`None`** for production.

---

### Phase C — Behaviour modelling (consumes voter clustering outputs)

After Phase B, defaults relative to **`Dissertation/`** are:

- `Voter_Clustering/data/processed/master_votes_with_dao_cluster.parquet`
- `Voter_Clustering/outputs/voter_clustering/voter_cluster_assignments.csv`

Run:

```bash
cd Dissertation/behaviour_modelling
python run_pipeline.py
```

Override paths if files live elsewhere (e.g. large disks):

```bash
export BM_MASTER_VOTES_PARQUET=/path/to/master_votes_with_dao_cluster.parquet
export BM_VOTER_CLUSTER_ASSIGNMENTS_CSV=/path/to/voter_cluster_assignments.csv
export BM_OUTPUT_DIR=/path/to/behaviour_outputs
python run_pipeline.py
```

Outputs default to **`behaviour_modelling/outputs/behaviour_modelling/`** (dataset CSV, `agent2_artifacts/` checkpoint + tokenizer, `eval/` metrics). Behaviour modelling pulls Hugging Face pretrained weights — **first run needs internet** unless your student configures mirrors or offline caches (`HF_HOME` / `TRANSFORMERS_CACHE`).

---

## 5. Resources and runtime

- **Voter clustering**: CPU is fine; disk/RAM scale with Snapshot size; reserve space if `master_votes.parquet` reaches tens of GB.
- **Behaviour modelling**: Prefer GPU + CUDA PyTorch; CPU works but is slower. Use `--max_train_windows` and related flags for smaller trial runs (see `behaviour_modelling/train.py` and forwarded args in `test_pipeline/run_demo.py`).

---

## 6. Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| Cannot find `cluster_assignments.csv` | Wrong path or file not in repo | Ask student for absolute path; set `VOTER_DAO_CLUSTERS_CSV` |
| Empty or heavily skipped voter clustering | Too few rows per DAO slice in Mode A | Adjust `VOTER_CLUSTER_MIN_ROWS_MODE_A`, add data, or change `CLUSTERING_MODE` in `config.py` |
| Cannot download HF model | No outbound network / proxy | Use HF mirror, offline cache, or student-provided cache directory |
| Large files missing after `git clone` | Listed in `.gitignore` | Regenerate parquet and models on the server per this runbook |

---

## 7. Windows PowerShell — environment variables

```powershell
$env:VOTER_SNAPSHOT_ROOT="D:\data\snapshot_votes_441\spaces"
$env:VOTER_DAO_CLUSTERS_CSV="D:\Dissertation\DAO_Clustering_cleaning\outputs\cluster_results\no_outliers\cluster_assignments.csv"
```

---

## 8. Checklist — information to get from your student before running

1. Absolute path to the Snapshot vote root on the server.  
2. Path to `cluster_assignments.csv` if different from the default.  
3. Full Snapshot run vs trial with `MAX_PARQUET_FILES`.  
4. Whether behaviour modelling must be fully offline, and whether the default pretrained model name changes.  
5. Preferred output locations (e.g. `BM_OUTPUT_DIR` on a large volume).

---

For path or environment errors, share the **full traceback**, the **exact command**, and **relevant environment variables** with your student so they can cross-check `Voter_Clustering/RUN_ON_SERVER.txt` and `test_pipeline/README.md`.
