#!/usr/bin/env python3
"""
Metrics for Agent 2.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def macro_prf(y_true, y_pred) -> Dict[str,float]:
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}
