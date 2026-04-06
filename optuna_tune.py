import os
import sys
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
import argparse
import datetime
import mlflow

from config import Config
from dataset import BrainDataset, collate_fn
from model import build_model
from trainer import train_epoch, validate

def _discover_files(data_dir: str, session_glob: str, filename: str):
    pattern = os.path.join(data_dir, session_glob, filename)
    files = sorted(glob(pattern))
    return files

def objective(trial: optuna.Trial, model_type: str):
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
        # ── 0. Create a fresh config object for this trial ─────────────────────────
        trial_config = Config()
        
        # ── 1. Set fixed parameters ─────────────────────────────────────────────
        # epoch fixed to 30, early_stopping_patience = 10 (no tuning)
        trial_config.num_epochs = 30
        trial_config.early_stopping_patience = 10
        trial_config.model_type = model_type
        
        # ── 2. Suggest hyperparameters ─────────────────────────────────────────
        # common parameters
        trial_config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        trial_config.batch_size = trial.suggest_categorical("batch_size", [8, 16])

        if model_type == "Transformer":
            # d_model choices based on comment: 512, 768
            trial_config.d_model = trial.suggest_categorical("d_model", [512, 768, 1024])
            trial_config.nhead = trial.suggest_categorical("nhead", [8, 12])
            
            # NOTE: d_model must be divisible by nhead
            if trial_config.d_model % trial_config.nhead != 0:
                mlflow.set_tag("pruned", "invalid_architecture")
                raise optuna.exceptions.TrialPruned()
                
            trial_config.num_layers = trial.suggest_categorical("num_layers", [4, 6, 8])
            trial_config.dim_feedforward = trial.suggest_categorical("dim_feedforward", [1024, 2048])
            trial_config.dropout = trial.suggest_float("dropout", 0.1, 0.5)
            
        elif model_type == "LSTM":
            # LSTM specific params from comment
            trial_config.lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [512, 768, 1024])
            trial_config.lstm_num_layers = trial.suggest_categorical("lstm_num_layers", [3, 4, 5])
            trial_config.lstm_dropout = trial.suggest_categorical("lstm_dropout", [0.3, 0.4, 0.5])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        mlflow.log_params(trial_config.as_dict())
        mlflow.log_param("trial_number", trial.number)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 3. Data Loaders ───────────────────────────────────────────────────
    train_files = _discover_files(trial_config.DATA_DIR, trial_config.SESSION_GLOB, trial_config.TRAIN_FILENAME)
    val_files = _discover_files(trial_config.DATA_DIR, trial_config.SESSION_GLOB, trial_config.VAL_FILENAME)
    
    if not train_files:
        raise RuntimeError(f"No train files found under {trial_config.DATA_DIR}/{trial_config.SESSION_GLOB}/{trial_config.TRAIN_FILENAME}")
        
    train_dataset = BrainDataset(train_files, is_test=False, max_len=trial_config.max_seq_len, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=trial_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    if val_files:
        val_dataset = BrainDataset(val_files, is_test=False, max_len=trial_config.max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=trial_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    else:
        print(f"[Trial {trial.number}] WARNING: No val files found – using train loader for validation.")
        val_loader = train_loader

    # ── 4. Build Model, Optimizer, Scheduler ─────────────────────────────
    try:
        model = build_model(trial_config, device)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial_config.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=4, factor=0.2
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0

        # ── 5. Training Loop ─────────────────────────────────────────────────
        for epoch in range(1, trial_config.num_epochs + 1):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer,
                device, epoch, cfg=trial_config, mlflow_run=run
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion,
                device, epoch, cfg=trial_config, mlflow_run=run
            )

            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metric("lr", current_lr, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
            else:
                patience_counter += 1

            # Update accuracy in Optuna (shows in dashboard table)
            trial.set_user_attr("val_acc", val_acc)
            trial.set_user_attr("best_val_acc", best_val_acc)

            # Report intermediate objective value manually before checking for prune condition
            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                mlflow.set_tag("pruned", "true")
                raise optuna.exceptions.TrialPruned()

            if patience_counter >= trial_config.early_stopping_patience:
                # Early stopping triggered, end training for this trial
                mlflow.set_tag("early_stopped", "true")
                break

        mlflow.log_metric("final_best_val_loss", best_val_loss)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Trial {trial.number}] Pruning due to OOM error.")
            mlflow.set_tag("OOM", "true")
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    # Make sure indent closes with mlflow.start_run
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Brain-to-Text")
    # Using Transformer / LSTM based on provided --model flag. By default runs Transformer
    parser.add_argument("--model", type=str, default="Transformer", choices=["LSTM", "Transformer"],
                        help="Model type to tune")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    args = parser.parse_args()

    model_type = args.model
    study_name = f"{model_type.lower()}_optuna"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    print(f"[*] Starting Optuna hyperparameter tuning for {model_type}")
    print(f"[*] Max Epochs: 30 | Early Stopping: 10")
    print(f"[*] Storage: {storage_name}")
    print(f"[*] Tuning will run for {args.trials} trials.")
    
    # Setup MLFlow
    mlflow.set_tracking_uri(Config().MLFLOW_TRACKING_URI)
    experiment_name = f"{Config().MLFLOW_EXPERIMENT_NAME}_{model_type.lower()}_tuning"
    mlflow.set_experiment(experiment_name)
    
    run_name = f"optuna_study_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        with mlflow.start_run(run_name=run_name):
            study.optimize(lambda trial: objective(trial, model_type), n_trials=args.trials)
            
            # Log best overall params to the parent run
            if len(study.trials) > 0:
                try:
                    best_trial = study.best_trial
                    mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
                    mlflow.log_metric("overall_best_val_loss", best_trial.value)
                except ValueError:
                    pass
    except KeyboardInterrupt:
        print("\n[!] Tuning interrupted by user.")
        
    print("\n[+] Tuning complete!")
    print("[+] Best trial:")
    best_trial = study.best_trial
    print(f"  Value (val_loss): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
