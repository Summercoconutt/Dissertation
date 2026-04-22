#!/usr/bin/env python3
"""
Train behaviour model from built dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from load_data import load_dataset, normalise_columns, select_numeric_columns, split_by_voter
from windows import build_windows
from dataset import WindowDataset, collate_fn
from model import TimeSeriesClassifier
from metrics import macro_prf


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    n = counts.sum()
    w = n / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs/behaviour_modelling/agent2_artifacts")
    ap.add_argument("--pretrained", type=str, default="roberta-base")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_train_windows",
        type=int,
        default=0,
        help="If >0, randomly subsample training windows (smoke tests).",
    )
    ap.add_argument(
        "--max_valid_windows",
        type=int,
        default=0,
        help="If >0, randomly subsample validation windows (smoke tests).",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = normalise_columns(load_dataset(Path(args.dataset_csv)))
    train_df, valid_df = split_by_voter(df, train_frac=0.8, seed=args.seed)
    numeric_cols = select_numeric_columns()

    train_windows = build_windows(train_df, window_size=args.window, numeric_cols=numeric_cols)
    valid_windows = build_windows(valid_df, window_size=args.window, numeric_cols=numeric_cols)
    if not train_windows or not valid_windows:
        raise RuntimeError("Empty windows. Check data volume or reduce --window.")

    rng_sub = np.random.default_rng(args.seed)

    def _maybe_sub(ws: list, cap: int) -> list:
        if cap <= 0 or len(ws) <= cap:
            return ws
        idx = rng_sub.choice(len(ws), size=cap, replace=False)
        return [ws[i] for i in sorted(idx)]

    train_windows = _maybe_sub(train_windows, args.max_train_windows)
    valid_windows = _maybe_sub(valid_windows, args.max_valid_windows)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PREDICT]", "[LABEL_0]", "[LABEL_1]", "[LABEL_2]"]})

    train_ds = WindowDataset(train_windows, tokenizer, args.max_length)
    valid_ds = WindowDataset(valid_windows, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    feat_dim = len(train_windows[0].window_features[0])
    model = TimeSeriesClassifier(pretrained_model_name=args.pretrained, feat_dim=feat_dim).to(device)
    model.text_encoder.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    class_weights = compute_class_weights(np.array([w.target_label for w in train_windows], dtype=int)).to(device)

    best_f1 = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        tr_true, tr_pred, tr_loss = [], [], 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}"):
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(batch)
            loss = model.loss_fn(out["logits"], batch["labels"], class_weights=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += float(loss.item())
            tr_true.extend(batch["labels"].detach().cpu().numpy().tolist())
            tr_pred.extend(out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist())

        train_m = macro_prf(tr_true, tr_pred)

        model.eval()
        va_true, va_pred = [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"valid {epoch+1}/{args.epochs}"):
                for k in batch:
                    batch[k] = batch[k].to(device)
                out = model(batch)
                va_true.extend(batch["labels"].detach().cpu().numpy().tolist())
                va_pred.extend(out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist())
        valid_m = macro_prf(va_true, va_pred)
        print(
            f"epoch={epoch+1} loss={tr_loss/max(len(train_loader),1):.4f} "
            f"train_f1={train_m['f1']:.4f} valid_f1={valid_m['f1']:.4f}"
        )

        if valid_m["f1"] > best_f1:
            best_f1 = valid_m["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    outdir = Path(args.output_dir)
    tok_dir = outdir / "tokenizer"
    outdir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), outdir / "model.pt")
    tokenizer.save_pretrained(tok_dir)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pretrained": args.pretrained,
                "window": args.window,
                "max_length": args.max_length,
                "feat_dim": int(feat_dim),
                "best_valid_f1": float(best_f1),
                "label_map": {"FOR": 0, "AGAINST": 1, "ABSTAIN": 2},
            },
            f,
            indent=2,
        )
    print(f"saved model to: {outdir}")


if __name__ == "__main__":
    main()
