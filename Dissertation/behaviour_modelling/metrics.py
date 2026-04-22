#!/usr/bin/env python3
"""
Metrics utilities.
"""
from __future__ import annotations

from typing import Dict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def macro_prf(y_true, y_pred) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f), "accuracy": float(accuracy_score(y_true, y_pred))}
