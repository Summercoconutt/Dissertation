#!/usr/bin/env python3
"""
Dataset and collate for Agent 2.
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import numpy as np

@dataclass
class EncodedWindow:
    input_ids: torch.Tensor        # (W, L)
    attention_mask: torch.Tensor   # (W, L)
    num_feats: torch.Tensor        # (W, F)
    label: int                     # scalar
    voter_id: str
    cluster_id: int

class WindowDataset(Dataset):
    def __init__(self, windows: List[Any], tokenizer, max_length: int = 128):
        self.windows = windows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> EncodedWindow:
        w = self.windows[idx]
        # Encode each step text separately
        ids, masks = [], []
        for t in w.window_texts:
            enc = self.tok(
                t,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            ids.append(enc["input_ids"][0])          # (L,)
            masks.append(enc["attention_mask"][0])   # (L,)
        input_ids = torch.stack(ids, dim=0)          # (W, L)
        attention_mask = torch.stack(masks, dim=0)   # (W, L)
        # Numeric features
        num_feats = torch.tensor(np.stack(w.window_features, axis=0), dtype=torch.float32)  # (W, F)
        return EncodedWindow(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_feats=num_feats,
            label=int(w.target_label),
            voter_id=w.voter_id,
            cluster_id=int(w.cluster_id)
        )

def collate_fn(batch: List[EncodedWindow]) -> Dict[str, torch.Tensor]:
    # stack by dimension
    input_ids = torch.stack([b.input_ids for b in batch], dim=0)            # (B, W, L)
    attention_mask = torch.stack([b.attention_mask for b in batch], dim=0)  # (B, W, L)
    num_feats = torch.stack([b.num_feats for b in batch], dim=0)            # (B, W, F)
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)       # (B,)
    clusters = torch.tensor([b.cluster_id for b in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_feats": num_feats,
        "labels": labels,
        "clusters": clusters
    }
