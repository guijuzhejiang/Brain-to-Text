"""
model.py – BrainTransformer architecture

Components:
    PositionalEncoding  – sinusoidal positional encoding
    BrainTransformer    – Transformer encoder that maps neural signals → class logits
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from config import config

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) – batch_first friendly
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class BrainTransformer(nn.Module):
    """
    Transformer-encoder model for brain-to-text decoding.

    Args:
        cfg: a Config instance (defaults to the singleton `config`).
    """

    def __init__(self, cfg=config) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_size, cfg.d_model)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.num_layers)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)
        src_key_padding_mask : (batch, seq_len) bool – True = ignore

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        x = self.input_proj(x)           # → (B, T, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_proj(x)     # → (B, T, vocab_size)
        return logits

def build_model(cfg=config, device: torch.device | str = "cpu") -> BrainTransformer:
    model = BrainTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[BrainTransformer] Parameters: {n_params:,}")
    return model
