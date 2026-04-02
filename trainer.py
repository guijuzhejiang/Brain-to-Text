"""
trainer.py – Training / validation pipeline with MLFlow tracking

Public API
----------
    train_epoch(model, dataloader, criterion, optimizer, device, epoch, run)
    validate(model, dataloader, criterion, device, epoch, run)
    save_checkpoint(model, optimizer, epoch, loss, cfg)
    generate_submission(model, test_files, output_file, device, cfg)
    generate_simple_submission(model, test_file, output_file, device, cfg)
"""

from __future__ import annotations

import os
from typing import List

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import mlflow
import mlflow.pytorch

from config import config
from dataset import BrainDataset, collate_fn
from torch.utils.data import DataLoader


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        cfg=config,
        mlflow_run=None,
) -> float:
    """Run one training epoch and return the average token-level loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}")
    for batch_idx, (neural_data, labels, padding_mask, _) in enumerate(pbar):
        neural_data = neural_data.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)

        optimizer.zero_grad()

        logits = model(neural_data, src_key_padding_mask=padding_mask)

        batch_size, neural_len, vocab_size = logits.shape
        label_len = labels.shape[1]

        # Align logits and labels along the time axis
        if neural_len >= label_len:
            logits_aligned = logits[:, :label_len, :]
        else:
            logits_aligned = F.pad(logits, (0, 0, 0, label_len - neural_len))

        logits_flat = logits_aligned.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        batch_tokens = (labels_flat != 0).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        avg = total_loss / total_tokens if total_tokens > 0 else 0.0
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg:.4f}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    # ---------- MLFlow logging ----------
    if mlflow_run is not None:
        mlflow.log_metric("train/loss", avg_loss, step=epoch)

    return avg_loss


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        cfg=config,
        mlflow_run=None,
):
    """Evaluate on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for neural_data, labels, padding_mask, _ in dataloader:
            neural_data = neural_data.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(neural_data, src_key_padding_mask=padding_mask)

            batch_size, neural_len, vocab_size = logits.shape
            label_len = labels.shape[1]

            if neural_len >= label_len:
                logits_aligned = logits[:, :label_len, :]
            else:
                logits_aligned = F.pad(logits, (0, 0, 0, label_len - neural_len))

            logits_flat = logits_aligned.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)

            loss = criterion(logits_flat, labels_flat)

            preds = torch.argmax(logits_flat, dim=1)
            mask = labels_flat != 0
            correct += ((preds == labels_flat) & mask).sum().item()
            batch_tokens = mask.sum().item()

            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = correct / total_tokens if total_tokens > 0 else 0.0

    # ---------- MLFlow logging ----------
    if mlflow_run is not None:
        mlflow.log_metric("val/loss", avg_loss, step=epoch)
        mlflow.log_metric("val/accuracy", accuracy, step=epoch)

    return avg_loss, accuracy


def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        cfg=config,
) -> str:
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(cfg.CHECKPOINT_DIR, cfg.BEST_MODEL_NAME)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    return path


def generate_submission(
        model: nn.Module,
        test_files: List[str],
        output_file: str,
        device: torch.device,
        cfg=config,
) -> None:
    """Batch-mode prediction; recommended for speed."""
    model.eval()

    test_dataset = BrainDataset(test_files, is_test=True, max_len=cfg.max_seq_len)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )

    trial_ids: List[str] = []
    predictions: List[str] = []

    with torch.no_grad():
        for neural_data, _, padding_mask, trial_keys in tqdm(
                test_loader, desc="[Submission] Batch prediction"
        ):
            neural_data = neural_data.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(neural_data, src_key_padding_mask=padding_mask)
            preds = torch.argmax(logits[:, : cfg.max_seq_len, :], dim=-1)

            for i, key in enumerate(trial_keys):
                seq = preds[i].cpu().numpy()
                if len(seq) > cfg.max_seq_len:
                    seq = seq[: cfg.max_seq_len]
                elif len(seq) < cfg.max_seq_len:
                    seq = np.pad(seq, (0, cfg.max_seq_len - len(seq)))

                trial_ids.append(key)
                predictions.append(" ".join(map(str, seq)))

    _write_submission(trial_ids, predictions, output_file)


def generate_simple_submission(
        model: nn.Module,
        test_file: str,
        output_file: str,
        device: torch.device,
        cfg=config,
) -> None:
    """Trial-by-trial fallback; slower but simpler."""
    model.eval()

    with h5py.File(test_file, "r") as f:
        trial_ids_raw = list(f.keys())

    trial_ids: List[str] = []
    predictions: List[str] = []

    with torch.no_grad():
        for trial_id in tqdm(trial_ids_raw, desc="[Submission] Trial-by-trial"):
            with h5py.File(test_file, "r") as f:
                neural_data = (
                    torch.tensor(f[trial_id]["input_features"][:], dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )

            max_len = neural_data.shape[1]
            padding_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)

            logits = model(neural_data, src_key_padding_mask=padding_mask)
            seq = torch.argmax(logits[0, : cfg.max_seq_len, :], dim=-1).cpu().numpy()

            if len(seq) > cfg.max_seq_len:
                seq = seq[: cfg.max_seq_len]
            elif len(seq) < cfg.max_seq_len:
                seq = np.pad(seq, (0, cfg.max_seq_len - len(seq)))

            trial_ids.append(trial_id)
            predictions.append(" ".join(map(str, seq)))

    _write_submission(trial_ids, predictions, output_file)


def _write_submission(
        trial_ids: List[str], predictions: List[str], output_file: str
) -> None:
    df = pd.DataFrame({"id": trial_ids, "predictions": predictions})
    df.to_csv(output_file, index=False)
    print(f"[Submission] Saved → {output_file}")
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print(df.head())
