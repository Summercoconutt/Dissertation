#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Agent 2 on the validation split and export paper-ready tables/plots.
- Auto-discovers agent2_artifacts (or use --artifacts_dir)
- Rebuilds per-voter validation windows with the same seed as training
- Saves blue-themed figures and LaTeX tables
- NEW: Macro-level summary table (overall + by cluster) and per-cluster confusion matrices

Label mapping is FIXED: For=0, Against=1, Abstain=2.
"""

import argparse
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", type=str, default="",
                    help="Folder containing agent2_fixed code (dataset.py, model.py, ...). If empty, use the script's parent.")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Directory where cluster_0/1/2_dataset.csv reside.")
    ap.add_argument("--artifacts_dir", type=str, default="",
                    help="Directory of agent2_artifacts (config.json + agent2_model.pt). If empty, auto-discover.")
    ap.add_argument("--output_dir", type=str, default="",
                    help="Where to save eval outputs. Default: <project_root>/agent2_eval (project_root=artifacts parent)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for split_by_voter (must match training).")
    ap.add_argument("--batch_size", type=int, default=64)
    return ap.parse_args()


# =========================
# Utilities
# =========================
def autodiscover_artifacts(root_candidates):
    """
    Try several likely locations; if not found, do a light recursive search under /content and Drive.
    """
    for cand in root_candidates:
        c = Path(cand)
        if (c / "config.json").exists() and (c / "agent2_model.pt").exists():
            return c

    # Light recursive search (avoid scanning entire drive)
    search_roots = [Path("/content"), Path("/content/drive/MyDrive")]
    hits = []
    for root in search_roots:
        try:
            for p in root.rglob("agent2_artifacts/config.json"):
                if (p.parent / "agent2_model.pt").exists():
                    hits.append(p.parent)
        except Exception:
            pass
    return hits[0] if hits else None


def plot_confusion(cm, out_path_png, title="Confusion Matrix"):
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1, 2], ["For(0)", "Against(1)", "Abstain(2)"], rotation=35, ha="right")
    plt.yticks([0, 1, 2], ["For(0)", "Against(1)", "Abstain(2)"])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path_png, dpi=300); plt.close()


# =========================
# Main
# =========================
def main():
    args = parse_args()

    # Locate code dir with agent2_fixed modules
    script_dir = Path(__file__).resolve().parent
    code_dir = Path(args.code_dir) if args.code_dir else script_dir
    sys.path.insert(0, str(code_dir))

    # Import project modules (must exist in code_dir)
    from load_data import load_and_merge, normalise_columns, select_numeric_columns, split_by_voter
    from windows import build_windows
    from dataset import WindowDataset, collate_fn
    from model import TimeSeriesClassifier
    from transformers import AutoTokenizer

    # -------------------------
    # Data loading
    # -------------------------
    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"data_dir not found: {data_dir}"

    # Robust loader: prefer load_and_merge([dir]) that globs cluster_* files
    try:
        df = load_and_merge([data_dir])
    except Exception:
        # fallback to explicit file list
        csvs = [data_dir / f"cluster_{i}_dataset.csv" for i in range(3)]
        df = load_and_merge(csvs)

    df = normalise_columns(df)
    # Keep only valid labels (For=0, Against=1, Abstain=2)
    df = df[df["label_id"].isin([0, 1, 2])].copy()

    # -------------------------
    # Artifacts (config / model / tokenizer)
    # -------------------------
    if args.artifacts_dir:
        arti_dir = Path(args.artifacts_dir)
        if not ((arti_dir / "config.json").exists() and (arti_dir / "agent2_model.pt").exists()):
            raise FileNotFoundError(f"--artifacts_dir provided but missing files: {arti_dir}")
    else:
        # Candidates: common relative positions from this script/code_dir
        candidates = [
            script_dir / "agent2_artifacts",
            code_dir / "agent2_artifacts",
            Path.cwd() / "agent2_artifacts",
            script_dir.parent / "agent2_artifacts",
            script_dir.parent / "agent2_fixed" / "agent2_artifacts",
        ]
        arti_dir = autodiscover_artifacts(candidates)
        if arti_dir is None:
            raise FileNotFoundError(
                "Cannot locate agent2_artifacts. Provide --artifacts_dir or keep default layout "
                "(agent2_fixed/agent2_artifacts with config.json & agent2_model.pt)."
            )

    print("Using artifacts at:", arti_dir)

    # Read config
    cfg = json.loads((arti_dir / "config.json").read_text())
    # Fixed mapping requirement
    expected_map = {"FOR": 0, "AGAINST": 1, "ABSTAIN": 2}
    if cfg.get("label_map") != expected_map:
        raise ValueError(f"Label mapping mismatch in config.json: {cfg.get('label_map')} vs expected {expected_map}")

    # Load tokenizer FIRST (contains 4 added special tokens), then build model and RESIZE embeddings
    tokenizer = AutoTokenizer.from_pretrained(arti_dir / "tokenizer")

    model = TimeSeriesClassifier(
        pretrained_model_name=cfg["pretrained"],
        feat_dim=cfg["feat_dim"]
    )
    # Critical: resize embeddings BEFORE load_state_dict
    model.text_encoder.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(arti_dir / "agent2_model.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    # -------------------------
    # Build validation windows (same split seed as training)
    # -------------------------
    train_df, valid_df = split_by_voter(df, train_frac=0.8, seed=args.seed)
    numeric_cols = select_numeric_columns(df)

    def build_windows_for(df_part):
        return build_windows(
            df=df_part,
            window_size=cfg["window"],
            text_col="text",
            label_col="label_id",
            voter_col="voter",
            time_col="vote_ts",
            cluster_col="cluster_id",
            numeric_cols=numeric_cols
        )

    valid_windows = build_windows_for(valid_df)
    if len(valid_windows) == 0:
        raise RuntimeError(
            "No validation windows built. "
            "Check window size W in config.json and your data coverage; try a smaller W or different seed."
        )

    valid_ds = WindowDataset(valid_windows, tokenizer, max_length=cfg["max_length"])
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # -------------------------
    # Output dir
    # -------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        # Default: <project_root>/agent2_eval (project_root = artifacts parent)
        out_dir = arti_dir.parent / "agent2_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving outputs to:", out_dir)

    # -------------------------
    # Inference
    # -------------------------
    model.eval()
    y_true, y_pred = [], []
    probs_all = []
    clusters = [w.cluster_id for w in valid_windows]

    with torch.no_grad():
        for batch in valid_loader:
            for k in ["input_ids", "attention_mask", "num_feats", "labels", "clusters"]:
                batch[k] = batch[k].to(device)
            out = model(batch)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs)
            y_true.extend(batch["labels"].cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    probs_all = np.vstack(probs_all)
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    clusters = np.array(clusters, dtype=int)

    # -------------------------
    # Reports & confusion matrix (overall)
    # -------------------------
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0,
        target_names=["For(0)", "Against(1)", "Abstain(2)"]
    )
    rep_df = pd.DataFrame(report).T
    rep_df.to_csv(out_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm,
                         index=["For(0)", "Against(1)", "Abstain(2)"],
                         columns=["For(0)", "Against(1)", "Abstain(2)"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm_norm, index=cm_df.index, columns=cm_df.columns).to_csv(out_dir / "confusion_matrix_normalised.csv")

    plot_confusion(cm, out_dir / "confusion_matrix_blue.png", title="Confusion Matrix (Overall)")

    # Overall metrics for macro-by-cluster table
    overall_macro_p = report["macro avg"]["precision"]
    overall_macro_r = report["macro avg"]["recall"]
    overall_macro_f1 = report["macro avg"]["f1-score"]
    overall_acc = accuracy_score(y_true, y_pred)

    # -------------------------
    # ECE & reliability curves (overall, per class)
    # -------------------------
    def ece_curve(y_true_, probs_, cls, n_bins=10):
        p = probs_[:, cls]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(p, bins, right=True) - 1
        acc, conf, cnt = [], [], []
        for b in range(n_bins):
            m = (idx == b)
            if m.sum() == 0:
                acc.append(np.nan); conf.append(np.nan); cnt.append(0)
            else:
                acc.append((y_true_[m] == cls).mean())
                conf.append(p[m].mean())
                cnt.append(int(m.sum()))
        acc, conf, cnt = np.array(acc), np.array(conf), np.array(cnt)
        valid = cnt > 0
        ece = np.sum((np.abs(acc[valid] - conf[valid]) * cnt[valid])) / max(cnt.sum(), 1)
        return bins, acc, conf, cnt, float(ece)

    names = ["For", "Against", "Abstain"]
    ece_rows = []
    for c in [0, 1, 2]:
        bins, acc, conf, cnt, ece = ece_curve(y_true, probs_all, c, n_bins=10)
        ece_rows.append({"class": names[c], "ece": ece})
        bc = (bins[:-1] + bins[1:]) / 2.0
        # Reliability (blue)
        plt.figure(figsize=(5.2, 4.0))
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.plot(bc, acc, marker="o", linewidth=1.5, color="tab:blue")
        plt.xlabel("Predicted probability"); plt.ylabel("Empirical accuracy")
        plt.title(f"Reliability — {names[c]} (ECE={ece:.3f})")
        plt.ylim(0, 1); plt.xlim(0, 1); plt.grid(alpha=.2)
        plt.tight_layout(); plt.savefig(out_dir / f"reliability_{names[c].lower()}_blue.png", dpi=300); plt.close()
        # Probability histogram (blue)
        plt.figure(figsize=(5.2, 3.6))
        plt.hist(probs_all[:, c], bins=20, color="tab:blue")
        plt.xlabel("Predicted probability"); plt.ylabel("Count")
        plt.title(f"Predicted proba histogram — {names[c]}")
        plt.tight_layout(); plt.savefig(out_dir / f"prob_hist_{names[c].lower()}_blue.png", dpi=300); plt.close()

    pd.DataFrame(ece_rows).to_csv(out_dir / "prob_calibration_ece.csv", index=False)

    # -------------------------
    # BY-CLUSTER: metrics + confusion matrices
    # -------------------------
    rows_summary = []  # for macro_by_cluster.csv & latex
    rows_summary.append({
        "Model": "Overall",
        "Macro-F1": overall_macro_f1,
        "Macro-Precision": overall_macro_p,
        "Macro-Recall": overall_macro_r,
        "Accuracy": overall_acc
    })

    for c_id in sorted(np.unique(clusters).tolist()):
        mask = (clusters == c_id)
        y_t = y_true[mask]; y_p = y_pred[mask]
        if y_t.size == 0:
            # still record a row with NaNs
            rows_summary.append({"Model": f"Cluster {c_id}",
                                 "Macro-F1": np.nan, "Macro-Precision": np.nan,
                                 "Macro-Recall": np.nan, "Accuracy": np.nan})
            continue

        rep_c = classification_report(
            y_t, y_p, output_dict=True, zero_division=0,
            target_names=["For(0)", "Against(1)", "Abstain(2)"]
        )
        macro_p = rep_c["macro avg"]["precision"]
        macro_r = rep_c["macro avg"]["recall"]
        macro_f1 = rep_c["macro avg"]["f1-score"]
        acc = accuracy_score(y_t, y_p)

        rows_summary.append({
            "Model": f"Cluster {c_id}",
            "Macro-F1": macro_f1,
            "Macro-Precision": macro_p,
            "Macro-Recall": macro_r,
            "Accuracy": acc
        })

        # Confusion matrix (per cluster)
        cm_c = confusion_matrix(y_t, y_p, labels=[0, 1, 2])
        # Save CSV and normalised CSV
        pd.DataFrame(cm_c,
                     index=["For(0)", "Against(1)", "Abstain(2)"],
                     columns=["For(0)", "Against(1)", "Abstain(2)"]) \
          .to_csv(out_dir / f"confusion_matrix_cluster_{c_id}.csv")
        cm_c_norm = cm_c / np.maximum(cm_c.sum(axis=1, keepdims=True), 1)
        pd.DataFrame(cm_c_norm,
                     index=["For(0)", "Against(1)", "Abstain(2)"],
                     columns=["For(0)", "Against(1)", "Abstain(2)"]) \
          .to_csv(out_dir / f"confusion_matrix_cluster_{c_id}_normalised.csv")

        # Plot blue confusion
        plot_confusion(cm_c, out_dir / f"confusion_matrix_cluster_{c_id}_blue.png",
                       title=f"Confusion Matrix (Cluster {c_id})")

    # -------------------------
    # Macro-level summary table (CSV + LaTeX with booktabs)
    # -------------------------
    summary_df = pd.DataFrame(rows_summary, columns=["Model", "Macro-F1", "Macro-Precision", "Macro-Recall", "Accuracy"])
    summary_df.to_csv(out_dir / "macro_by_cluster.csv", index=False)

    with open(out_dir / "latex_macro_by_cluster.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Macro-level performance across clusters and overall voter base.}
\label{tab:agent2_summary}
\begin{tabular}{lcccc}
\toprule
Model & Macro-F1 & Macro-Precision & Macro-Recall & Accuracy \\
\midrule
""")
        for _, r in summary_df.iterrows():
            f.write(f"{r['Model']} & {r['Macro-F1']:.3f} & {r['Macro-Precision']:.3f} & {r['Macro-Recall']:.3f} & {r['Accuracy']:.3f} \\\\\n")
        f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")

    print("\nDone. Artifacts saved to:", out_dir)


if __name__ == "__main__":
    main()
