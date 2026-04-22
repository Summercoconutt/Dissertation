# Agent 2 (Fixed) — Voting Behaviour Prediction

This folder contains a clean, Colab-ready implementation of Agent 2 with the following fixes:

- **Correct label mapping**: `FOR=0`, `AGAINST=1`, `ABSTAIN=2` (from the `Vote Label` column).
- **Per-voter windowing** (no cross-voter leakage) with window size `W`.
- **Special tokens added and used**: `[PREDICT]`, `[LABEL_0]`, `[LABEL_1]`, `[LABEL_2]`, and the model's embeddings are resized accordingly.
- **Text projection layer** is registered in `__init__` (trainable), not created inside `forward`.
- **Grouped split by voter** (80/20 by default).
- **Class-weighted loss** (inverse-frequency from the training split).
- Minimal, leakage-safe numeric features per step (`Voting Power`, `VP Ratio (%)`, booleans, time features).

## Data

Place your three CSVs in a directory, e.g.
```
/content/data/cluster_0_dataset.csv
/content/data/cluster_1_dataset.csv
/content/data/cluster_2_dataset.csv
```

Each CSV must include at least:
- `Voter` or `voter`
- `Vote Timestamp` (ISO, UTC)
- `Vote Label` in {FOR, AGAINST, ABSTAIN}
- `Proposal Title`
- `Voting Power`, `VP Ratio (%)` (optional but recommended)
- `Is Whale`, `Aligned With Majority` (optional)

## Colab quickstart

```bash
%cd /content
!pip -q install -r /content/agent2_fixed/requirements.txt

!python /content/agent2_fixed/train_agent2.py   --data_dir /content/data   --window 5   --max_length 128   --batch_size 32   --epochs 6   --pretrained roberta-base
```

Artifacts will be saved to `agent2_artifacts/`:
- `agent2_model.pt`
- `tokenizer/`
- `config.json`
