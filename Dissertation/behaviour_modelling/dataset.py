#!/usr/bin/env python3
"""
Dataset and collate for behaviour modelling windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EncodedWindow:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    num_feats: torch.Tensor
    label: int
    dao_cluster: int
    voter_cluster: int


class WindowDataset(Dataset):
    def __init__(self, windows: List[Any], tokenizer, max_length: int = 128):
        self.windows = windows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> EncodedWindow:
        w = self.windows[idx]
        ids, masks = [], []
        for text in w.window_texts:
            enc = self.tok(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            ids.append(enc["input_ids"][0])
            masks.append(enc["attention_mask"][0])
        return EncodedWindow(
            input_ids=torch.stack(ids, dim=0),
            attention_mask=torch.stack(masks, dim=0),
            num_feats=torch.tensor(np.stack(w.window_features, axis=0), dtype=torch.float32),
            label=int(w.target_label),
            dao_cluster=int(w.dao_cluster),
            voter_cluster=int(w.voter_cluster),
        )


def collate_fn(batch: List[EncodedWindow]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b.input_ids for b in batch], dim=0),
        "attention_mask": torch.stack([b.attention_mask for b in batch], dim=0),
        "num_feats": torch.stack([b.num_feats for b in batch], dim=0),
        "labels": torch.tensor([b.label for b in batch], dtype=torch.long),
        "dao_clusters": torch.tensor([b.dao_cluster for b in batch], dtype=torch.long),
        "voter_clusters": torch.tensor([b.voter_cluster for b in batch], dtype=torch.long),
    }
