#!/usr/bin/env python3
"""
Time-series classifier for Agent 2.
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel

class TimeSeriesClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_name: str = "roberta-base",
                 hidden_dim: int = 256,
                 feat_dim: int = 8,
                 label_emb_dim: int = 128,
                 pos_emb_dim: int = 128,
                 num_heads: int = 8,
                 ff_dim: int = 1024,
                 num_classes: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = hidden_dim
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.num_classes = num_classes
        # label/position embeddings
        self.label_emb = nn.Embedding(num_classes+1, label_emb_dim)  # +1 for current-step placeholder
        self.pos_emb = nn.Embedding(512, pos_emb_dim)                # support up to W<=512
        # numeric projection
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        fused_dim = hidden_dim + label_emb_dim + pos_emb_dim + 128
        encoder_layer = nn.TransformerEncoderLayer(d_model=fused_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=ff_dim,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        # temperature (optional calibration)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch: input_ids (B,W,L), attention_mask (B,W,L), num_feats (B,W,F), labels (B,), clusters (B,)
        """
        input_ids = batch["input_ids"]      # (B,W,L)
        attn = batch["attention_mask"]      # (B,W,L)
        feats = batch["num_feats"]          # (B,W,F)
        B,W,L = input_ids.size()

        # flatten steps for text encoding
        x = input_ids.view(B*W, L)
        a = attn.view(B*W, L)
        enc = self.text_encoder(input_ids=x, attention_mask=a)
        cls = enc.last_hidden_state[:,0,:]                      # (B*W, Htext)
        cls = self.text_proj(cls)                               # (B*W, hidden)
        cls = cls.view(B, W, -1)                                # (B,W,hidden)

        # label & position embeddings
        # Positions: 0..W-1
        pos_ids = torch.arange(W, device=input_ids.device).unsqueeze(0).expand(B, W)
        pos = self.pos_emb(pos_ids)                             # (B,W,pos_dim)

        # For label embeddings, we do not have explicit per-step labels in input pipeline;
        # use zeros for current step and historical steps can be inferred from text tag, so here keep zeros.
        label_tokens = torch.zeros((B,W), dtype=torch.long, device=input_ids.device)
        lab = self.label_emb(label_tokens)                      # (B,W,label_dim)

        # numeric projection
        fproj = self.feat_proj(feats)                           # (B,W,128)

        fused = torch.cat([cls, lab, pos, fproj], dim=-1)       # (B,W,fused)
        seq = self.temporal(fused)                              # (B,W,fused)
        last = seq[:,-1,:]                                      # (B,fused)
        logits = self.cls_head(last)                            # (B,C)
        return {"logits": logits}

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, class_weights: Optional[torch.Tensor]=None):
        if class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # temperature scaling
        return logits / self.temperature.clamp(min=1e-3)
