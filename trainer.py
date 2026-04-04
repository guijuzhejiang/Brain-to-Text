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
            logits_aligned = F.pad(logits, (0, 0, 0, label_len - neural_len), value=0)

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
                logits_aligned = F.pad(logits, (0, 0, 0, label_len - neural_len), value=0)

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
        val_accuracy: float,
        cfg=config,
        run_dir: str | None = None,
) -> str:
    """
    Save model checkpoint.

    Creates a timestamped sub-directory under cfg.CHECKPOINT_DIR so that
    different experimental runs never overwrite each other.

    Args:
        run_dir: If provided (e.g. from MLFlow run-id), use this name instead
                 of generating a timestamp.

    Returns:
        model_path – absolute strings.
    """
    import datetime

    if run_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"run_{timestamp}"

    checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, run_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, cfg.BEST_MODEL_NAME)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            'accuracy': val_accuracy,
            'config': config.__dict__
        },
        model_path,
    )
    print(f"[Checkpoint] Saved → {model_path}")
    return model_path


def generate_submission(
        model: nn.Module,
        test_files: List[str] | str,
        output_file: str,
        device: torch.device,
        cfg=config,
) -> None:
    """Entry point for submission generation. Selects method based on model_type."""
    if isinstance(test_files, list):
        test_file = test_files[0]
    else:
        test_file = test_files

    if getattr(cfg, 'model_type', 'LSTM') == 'Transformer':
        generate_submission_transformer(model, test_file, output_file, device, cfg)
    else:
        generate_submission_lstm(model, test_file, output_file, device, cfg)


def generate_submission_transformer(
        model: nn.Module,
        test_file: str,
        output_file: str,
        device: torch.device,
        cfg=config,
) -> None:
    """Transformer submission generation with batching."""
    from torch.nn.utils.rnn import pad_sequence
    model.eval()

    with h5py.File(test_file, 'r') as f:
        trial_ids = list(f.keys())

    print(f"[Submission] Found {len(trial_ids)} trials in test file")

    predictions = []
    batch_size = cfg.batch_size
    num_batches = (len(trial_ids) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(trial_ids))
            batch_trial_ids = trial_ids[start_idx:end_idx]

            batch_neural_data = []
            batch_lengths = []

            for trial_id in batch_trial_ids:
                with h5py.File(test_file, 'r') as f:
                    trial = f[trial_id]
                    neural_data = torch.tensor(trial['input_features'][:], dtype=torch.float32)
                    batch_neural_data.append(neural_data)
                    batch_lengths.append(len(neural_data))

            if batch_neural_data:
                neural_padded = pad_sequence(batch_neural_data, batch_first=True)
                max_len = neural_padded.shape[1]
                padding_mask = torch.arange(max_len).expand(len(batch_lengths), max_len) >= torch.tensor(
                    batch_lengths).unsqueeze(1)

                neural_padded = neural_padded.to(device)
                padding_mask = padding_mask.to(device)

                logits = model(neural_padded, src_key_padding_mask=padding_mask)
                batch_preds = torch.argmax(logits[:, :cfg.max_seq_len, :], dim=-1)

                for i, trial_id in enumerate(batch_trial_ids):
                    pred_sequence = batch_preds[i].cpu().numpy()

                    if len(pred_sequence) > cfg.max_seq_len:
                        pred_sequence = pred_sequence[:cfg.max_seq_len]
                    elif len(pred_sequence) < cfg.max_seq_len:
                        pred_sequence = np.pad(pred_sequence, (0, cfg.max_seq_len - len(pred_sequence)),
                                               mode='constant')

                    pred_str = ' '.join(map(str, pred_sequence))
                    predictions.append((trial_id, pred_str))

    _write_submission([p[0] for p in predictions], [p[1] for p in predictions], output_file, cfg)


def generate_submission_lstm(
        model: nn.Module,
        test_file: str,
        output_file: str,
        device: torch.device,
        cfg=config,
) -> None:
    """LSTM submission generation (trial-by-trial)."""
    model.eval()

    with h5py.File(test_file, 'r') as f:
        available_trials = list(f.keys())

    print(f"[Submission] Found {len(available_trials)} trials in test file")

    predictions = []

    with torch.no_grad():
        for trial_id in tqdm(available_trials, desc="Processing test trials"):
            with h5py.File(test_file, 'r') as f:
                trial = f[trial_id]
                neural_data = torch.tensor(trial['input_features'][:], dtype=torch.float32).unsqueeze(0).to(device)
                length = torch.tensor([neural_data.shape[1]], dtype=torch.long)

            # Using unified model forward with padding mask logic for LSTM (since we modified it earlier)
            # length is not used in our unified BrainLSTM forward signature directly, it uses src_key_padding_mask
            # Actually, to align with our unified model API:
            padding_mask = torch.zeros(1, neural_data.shape[1], dtype=torch.bool).to(device)
            logits = model(neural_data, src_key_padding_mask=padding_mask)
            
            preds = torch.argmax(logits[0, :cfg.max_seq_len, :], dim=-1).cpu().numpy()

            if len(preds) > cfg.max_seq_len:
                preds = preds[:cfg.max_seq_len]
            elif len(preds) < cfg.max_seq_len:
                preds = np.pad(preds, (0, cfg.max_seq_len - len(preds)), mode='constant')

            pred_str = ' '.join(map(str, preds))
            predictions.append((trial_id, pred_str))

    _write_submission([p[0] for p in predictions], [p[1] for p in predictions], output_file, cfg)


def _write_submission(
        trial_ids: List[str], predictions: List[str], output_file: str, cfg=config
) -> None:
    df = pd.DataFrame({"id": trial_ids, "predictions": predictions})
    
    # Ensure exactly expected_test_samples by injecting missing trial IDs
    if len(df) != cfg.expected_test_samples:
        existing_ids = set(df["id"].values)
        missing_rows = []
        placeholder_pred = " ".join(["0"] * cfg.max_seq_len)
        
        for i in range(cfg.expected_test_samples):
            trial_id = f"trial_{i:04d}"
            if trial_id not in existing_ids:
                missing_rows.append({"id": trial_id, "predictions": placeholder_pred})
                
        if missing_rows:
            missing_df = pd.DataFrame(missing_rows)
            df = pd.concat([df, missing_df], ignore_index=True)
            
        # Final safety truncation if still too long
        if len(df) > cfg.expected_test_samples:
            df = df.head(cfg.expected_test_samples)
            
    # Sort by trial id to ensure order
    df = df.sort_values('id').reset_index(drop=True)

    df.to_csv(output_file, index=False)
    print(f"[Submission] Saved → {output_file}")
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print(df.head())
