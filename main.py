"""
main.py – Brain-to-Text training entry point

Usage:
    conda run -n py312_cu121 python main.py

Steps:
    1. Discover all HDF5 train/val/test files under config.DATA_DIR
    2. Build dataset and dataloaders
    3. Build model, optimizer, scheduler
    4. Run training loop inside an MLFlow run
    5. Log params + metrics; register best model in MLFlow Model Registry
    6. Generate submission CSV from test data
"""

from __future__ import annotations

import os
import sys
from glob import glob

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

from config import config
from dataset import BrainDataset, collate_fn, explore_files
from model import build_model
from trainer import (
    train_epoch,
    validate,
    save_checkpoint,
    generate_submission,
)
from torch.utils.data import DataLoader
import datetime


def _discover_files(data_dir: str, session_glob: str, filename: str):
    """Return sorted list of HDF5 paths matching the glob pattern."""
    pattern = os.path.join(data_dir, session_glob, filename)
    files = sorted(glob(pattern))
    return files


def main():
    # ── 0. Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")

    # ── 1. Discover files ─────────────────────────────────────────────────────
    train_files = _discover_files(config.DATA_DIR, config.SESSION_GLOB, config.TRAIN_FILENAME)
    val_files = _discover_files(config.DATA_DIR, config.SESSION_GLOB, config.VAL_FILENAME)
    test_files = _discover_files(config.DATA_DIR, config.SESSION_GLOB, config.TEST_FILENAME)

    print(f"\n[main] Found {len(train_files)} train file(s)")
    print(f"[main] Found {len(val_files)}   val   file(s)")
    print(f"[main] Found {len(test_files)}  test  file(s)")

    if not train_files:
        sys.exit(
            f"[ERROR] No train files found under {config.DATA_DIR}/{config.SESSION_GLOB}/"
            f"{config.TRAIN_FILENAME}\nPlease update config.DATA_DIR."
        )

    # ── 1a. Data exploration ───────────────────────────────────────────────────
    explore_files(train_files, n_preview=3)

    # ── 2. Datasets & DataLoaders ─────────────────────────────────────────────
    train_dataset = BrainDataset(train_files, is_test=False, max_len=config.max_seq_len, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # Use val files when available; fall back to train for quick sanity checks
    if val_files:
        val_dataset = BrainDataset(val_files, is_test=False, max_len=config.max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    else:
        print("[main] WARNING: No val files found – using train loader for validation.")
        val_loader = train_loader

    # ── 3. Model / optimizer / scheduler ─────────────────────────────────────
    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=config.num_epochs
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.2
    )

    # ── 4. MLFlow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # ── 5. Training loop (inside MLFlow run) ──────────────────────────────────
    run_name = f"run_{config.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n[MLFlow] Run ID : {run.info.run_id}")
        print(f"[MLFlow] Experiment: {config.MLFLOW_EXPERIMENT_NAME}")

        # --- Log all config hyper-parameters ---
        mlflow.log_params(
            {
                "input_size": config.input_size,
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_layers": config.num_layers,
                "dim_feedforward": config.dim_feedforward,
                "dropout": config.dropout,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "max_seq_len": config.max_seq_len,
                "grad_clip": config.grad_clip,
                "vocab_size": config.vocab_size,
                "num_train_files": len(train_files),
                "num_val_files": len(val_files),
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
            }
        )

        best_val_loss = float("inf")
        best_ckpt_path: str | None = None
        patience_counter = 0

        print("\n[main] Starting training...\n")
        for epoch in range(1, config.num_epochs + 1):
            print(f"── Epoch {epoch}/{config.num_epochs} ──")

            train_loss = train_epoch(
                model, train_loader, criterion, optimizer,
                device, epoch, config, run
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion,
                device, epoch, config, run
            )

            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metric("lr", current_lr, step=epoch)

            print(
                f"  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"lr={current_lr:.2e}"
            )

            # ── Checkpoint best model ──────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_ckpt_path = save_checkpoint(model, optimizer, epoch, val_loss, val_acc, config, run_dir=run_name)
                print(f"  ✓ Saved best checkpoint → {best_ckpt_path}  (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ⏳ Early stopping counter: {patience_counter}/{config.early_stopping_patience}")
            if patience_counter >= config.early_stopping_patience:
                print("🛑 Early stopping triggered!")
                break

        # ── 6. Log best metrics & register model ──────────────────────────
        mlflow.log_metric("best_val_loss", best_val_loss)

        if best_ckpt_path and os.path.exists(best_ckpt_path):
            # Load best weights before logging the model
            ckpt = torch.load(best_ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)

            # Log model artifact to MLFlow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=(
                    config.MLFLOW_MODEL_NAME if config.MLFLOW_MODEL_NAME else None
                ),
            )
            print(f"\n[MLFlow] Model logged"
                  + (f" & registered as '{config.MLFLOW_MODEL_NAME}'"
                     if config.MLFLOW_MODEL_NAME else ""))

        print("\n[main] Training complete!")

    # ── 7. Generate submission ─────────────────────────────────────────────────
    if test_files:
        print("\n[main] Generating submission...")
        generate_submission(
            model, test_files, config.SUBMISSION_FILE, device, config
        )
    else:
        print("\n[main] No test files found – creating dummy submission...")
        import pandas as pd
        dummy = {
            "id": [f"trial_{i:04d}" for i in range(59)],
            "predictions": [" ".join(["0"] * config.max_seq_len) for _ in range(59)],
        }
        pd.DataFrame(dummy).to_csv(config.SUBMISSION_FILE, index=False)
        print(f"  Dummy submission saved → {config.SUBMISSION_FILE}")

    print("\n[main] Done!  Verify submission columns: 'id' and 'predictions'.")


if __name__ == "__main__":
    main()
