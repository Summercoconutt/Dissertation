#!/usr/bin/env python3
"""
Evaluate trained behaviour model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer

from load_data import load_dataset, normalise_columns, select_numeric_columns, split_by_voter
from windows import build_windows
from dataset import WindowDataset, collate_fn
from model import TimeSeriesClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=str, required=True)
    ap.add_argument("--artifacts_dir", type=str, default="outputs/behaviour_modelling/agent2_artifacts")
    ap.add_argument("--output_dir", type=str, default="outputs/behaviour_modelling/eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--max_windows",
        type=int,
        default=0,
        help="If >0, randomly subsample validation windows (smoke tests).",
    )
    args = ap.parse_args()

    arti = Path(args.artifacts_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((arti / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(arti / "tokenizer")
    model = TimeSeriesClassifier(pretrained_model_name=cfg["pretrained"], feat_dim=cfg["feat_dim"])
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(arti / "model.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    df = normalise_columns(load_dataset(Path(args.dataset_csv)))
    _, valid_df = split_by_voter(df, train_frac=0.8, seed=args.seed)
    windows = build_windows(valid_df, window_size=cfg["window"], numeric_cols=select_numeric_columns())
    if not windows:
        raise RuntimeError("No validation windows for evaluation.")
    if args.max_windows > 0 and len(windows) > args.max_windows:
        rng = np.random.default_rng(args.seed + 1)
        idx = rng.choice(len(windows), size=args.max_windows, replace=False)
        windows = [windows[i] for i in sorted(idx)]

    ds = WindowDataset(windows, tokenizer, max_length=cfg["max_length"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    y_true, y_pred, dao, voter_c = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model(batch)["logits"]
            y_true.extend(batch["labels"].cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            dao.extend(batch["dao_clusters"].cpu().numpy().tolist())
            voter_c.extend(batch["voter_clusters"].cpu().numpy().tolist())

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out / "classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=["FOR", "AGAINST", "ABSTAIN"], columns=["FOR", "AGAINST", "ABSTAIN"]).to_csv(
        out / "confusion_matrix.csv"
    )

    detail = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "dao_cluster": dao, "voter_cluster": voter_c})
    detail.to_csv(out / "predictions_with_clusters.csv", index=False)

    # Simple macro-F1 by dao and voter cluster
    rows = []
    for key, group_col in [("dao_cluster", "dao_cluster"), ("voter_cluster", "voter_cluster")]:
        for gid, g in detail.groupby(group_col):
            if len(g) < 5:
                continue
            sub = classification_report(g["y_true"], g["y_pred"], output_dict=True, zero_division=0)
            rows.append({"group_type": key, "group_id": int(gid), "macro_f1": sub["macro avg"]["f1-score"], "n": len(g)})
    pd.DataFrame(rows).to_csv(out / "macro_f1_by_cluster.csv", index=False)
    print(f"saved evaluation to: {out}")


if __name__ == "__main__":
    main()
