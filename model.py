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

class BrainLSTM(nn.Module):
    """
    LSTM-based model for brain-to-text decoding.
    """
    def __init__(self, cfg=config) -> None:
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            batch_first=True,
            bidirectional=cfg.lstm_bidirectional,
            dropout=cfg.lstm_dropout if cfg.lstm_num_layers > 1 else 0
        )

        lstm_output_size = cfg.lstm_hidden_size * (2 if cfg.lstm_bidirectional else 1)
        self.output_proj = nn.Linear(lstm_output_size, cfg.vocab_size)
        self.dropout = nn.Dropout(cfg.lstm_dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengths = None
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1).cpu()

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(x)

        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.dropout(lstm_out)
        logits = self.output_proj(lstm_out)
        return logits

def build_model(cfg=config, device: torch.device | str = "cpu") -> nn.Module:
    if getattr(cfg, 'model_type', 'LSTM') == 'Transformer':
        model = BrainTransformer(cfg).to(device)
    else:
        model = BrainLSTM(cfg).to(device)
        
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{model.__class__.__name__}] Parameters: {n_params:,}")
    return model
