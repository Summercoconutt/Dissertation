#!/usr/bin/env python3
"""
Temporal classifier with text and numeric fusion.
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel


class TimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        hidden_dim: int = 256,
        feat_dim: int = 10,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, hidden_dim)

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        d_model = hidden_dim + 128
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_ids"]
        a = batch["attention_mask"]
        f = batch["num_feats"]
        bsz, steps, seqlen = x.size()

        enc = self.text_encoder(input_ids=x.view(bsz * steps, seqlen), attention_mask=a.view(bsz * steps, seqlen))
        cls = enc.last_hidden_state[:, 0, :]
        cls = self.text_proj(cls).view(bsz, steps, -1)
        f = self.feat_proj(f)
        fused = torch.cat([cls, f], dim=-1)
        seq = self.temporal(fused)
        logits = self.head(seq[:, -1, :])
        return {"logits": logits}

    def loss_fn(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if class_weights is None:
            return nn.CrossEntropyLoss()(logits, labels)
        return nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
