#!/usr/bin/env python3
"""
Training script for Agent 2.
Usage (Colab):
!pip -q install -r requirements.txt
!python train_agent2.py --data_dir /content/data --epochs 6 --window 5
"""
import argparse, os, math, time, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from load_data import load_and_merge, normalise_columns, select_numeric_columns, split_by_voter, VALID_LABELS
from windows import build_windows
from dataset import WindowDataset, collate_fn
from model import TimeSeriesClassifier
from metrics import macro_prf

def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    # inverse frequency
    counts = np.bincount(y, minlength=num_classes).astype(float)
    N = counts.sum()
    w = N / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing cluster_0/1/2_dataset.csv")
    ap.add_argument("--pretrained", type=str, default="roberta-base")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    csvs = [data_dir/f"cluster_{i}_dataset.csv" for i in range(3)]
    df = load_and_merge(csvs)
    df = normalise_columns(df)

    # filter to valid labels
    df = df[df["label_id"].isin([0,1,2])].copy()

    # split by voter
    train_df, valid_df = split_by_voter(df, train_frac=0.8, seed=args.seed)

    # tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    special_tokens = {"additional_special_tokens": ["[PREDICT]","[LABEL_0]","[LABEL_1]","[LABEL_2]"]}
    tokenizer.add_special_tokens(special_tokens)

    # numeric features to include per step
    numeric_cols = select_numeric_columns(df)

    # build windows per voter
    def build(df_part):
        return build_windows(
            df=df_part,
            window_size=args.window,
            text_col="text",
            label_col="label_id",
            voter_col="voter",
            time_col="vote_ts",
            cluster_col="cluster_id",
            numeric_cols=numeric_cols
        )
    train_windows = build(train_df)
    valid_windows = build(valid_df)

    # datasets
    train_ds = WindowDataset(train_windows, tokenizer, max_length=args.max_length)
    valid_ds = WindowDataset(valid_windows, tokenizer, max_length=args.max_length)

    # model
    # Determine numeric feat dim: feats length per step
    feat_dim = len(train_windows[0].window_features[0]) if len(train_windows)>0 else 8
    model = TimeSeriesClassifier(pretrained_model_name=args.pretrained, feat_dim=feat_dim)
    # resize token embeddings to include new tokens
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    model = model.cuda() if torch.cuda.is_available() else model

    # loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader)*args.epochs
    warmup_steps = int(0.1*total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # class weights from training labels
    y_train = np.array([w.target_label for w in train_windows], dtype=int)
    class_weights = compute_class_weights(y_train).to(next(model.parameters()).device)

    best_f1, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            for k in ["input_ids","attention_mask","num_feats","labels","clusters"]:
                batch[k] = batch[k].cuda() if torch.cuda.is_available() else batch[k]
            out = model(batch)
            loss = model.loss_fn(out["logits"], batch["labels"], class_weights=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            running_loss += float(loss.item())
            # predictions
            preds = out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].detach().cpu().numpy().tolist())
        train_metrics = macro_prf(y_true, y_pred)

        # validation
        model.eval()
        vy_true, vy_pred = [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [valid]"):
                for k in ["input_ids","attention_mask","num_feats","labels","clusters"]:
                    batch[k] = batch[k].cuda() if torch.cuda.is_available() else batch[k]
                out = model(batch)
                preds = out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist()
                vy_pred.extend(preds)
                vy_true.extend(batch["labels"].detach().cpu().numpy().tolist())
        val_metrics = macro_prf(vy_true, vy_pred)

        print(f"\nEpoch {epoch+1}: loss={running_loss/len(train_loader):.4f} "
              f"train F1={train_metrics['f1']:.4f} | valid F1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            print(f"✓ New best model: F1={best_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with F1={best_f1:.4f}")

    # save
    outdir = Path("agent2_artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir/"agent2_model.pt"
    tok_path = outdir/"tokenizer"
    tok_path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tok_path)
    with open(outdir/"config.json","w") as f:
        json.dump({
            "pretrained": args.pretrained,
            "window": args.window,
            "max_length": args.max_length,
            "feat_dim": int(feat_dim),
            "label_map": {"FOR":0,"AGAINST":1,"ABSTAIN":2},
            "best_f1": float(best_f1)
        }, f, indent=2)
    print(f"\nSaved model to {model_path} and tokenizer to {tok_path}")

if __name__ == "__main__":
    main()
