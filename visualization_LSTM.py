import os
import glob
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import config
from dataset import BrainDataset, collate_fn
from model import build_model
from main import _discover_files

def get_latest_checkpoint(checkpoint_dir=config.CHECKPOINT_DIR):
    """Find the most recently modified best_model.pth"""
    pattern = os.path.join(checkpoint_dir, "**", config.BEST_MODEL_NAME)
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        # Also check root of checkpoint dir
        pattern = os.path.join(checkpoint_dir, config.BEST_MODEL_NAME)
        checkpoints = glob.glob(pattern)
        
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    return latest_ckpt


def analyze_data_distribution(train_files):
    print("=== DATA DISTRIBUTION ANALYSIS ===")
    neural_lengths = []
    label_sequences = []

    for file_path in train_files:
        with h5py.File(file_path, 'r') as f:
            for trial_key in f.keys():
                trial = f[trial_key]
                neural_lengths.append(trial['input_features'].shape[0])
                if 'seq_class_ids' in trial:
                    label_sequences.append(trial['seq_class_ids'][:])
                else:
                    label_sequences.append(np.array([]))

    neural_lengths = np.array(neural_lengths)
    print(f"Neural sequence length statistics:")
    print(f"  Min: {neural_lengths.min() if len(neural_lengths) > 0 else 0}")
    print(f"  Max: {neural_lengths.max() if len(neural_lengths) > 0 else 0}")
    print(f"  Mean: {neural_lengths.mean():.1f}" if len(neural_lengths) > 0 else "  Mean: 0")
    print(f"  Std: {neural_lengths.std():.1f}" if len(neural_lengths) > 0 else "  Std: 0")

    plt.figure(figsize=(15, 12))
    plt.subplot(2, 3, 1)
    plt.hist(neural_lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Neural Sequence Lengths')
    plt.grid(True, alpha=0.3)

    all_labels = np.concatenate(label_sequences) if label_sequences else np.array([])
    label_counts = Counter(all_labels)

    plt.subplot(2, 3, 2)
    if label_counts:
        top_labels = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        plt.bar(top_labels.keys(), top_labels.values(), alpha=0.7)
        plt.xlabel('Label ID')
        plt.ylabel('Frequency')
        plt.title('Top 20 Most Frequent Labels')
        plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    label_lengths = [len(seq) for seq in label_sequences]
    plt.subplot(2, 3, 3)
    plt.hist(label_lengths, bins=20, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Label Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Label Sequence Lengths')
    plt.grid(True, alpha=0.3)
    
    return neural_lengths, label_sequences, all_labels, label_counts, label_lengths

def evaluate_model_performance(model, train_loader, device, neural_lengths):
    print("\n=== MODEL PERFORMANCE ANALYSIS ===")
    model.eval()
    sample_predictions = []
    sample_targets = []

    with torch.no_grad():
        batch_count = 0
        for batch in train_loader:
            if batch_count >= 3:
                break
            
            # collate_fn returns: neural_padded, labels, padding_mask, trial_keys
            neural_data, labels, padding_mask, _ = batch

            neural_data = neural_data.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(neural_data, src_key_padding_mask=padding_mask)

            batch_size, neural_len, vocab_size = logits.shape
            label_len = labels.shape[1]

            if neural_len >= label_len:
                logits_aligned = logits[:, :label_len, :]
            else:
                logits_aligned = logits

            preds = torch.argmax(logits_aligned, dim=-1)

            for i in range(batch_size):
                global_idx = batch_count * batch_size + i
                actual_neural_len = neural_lengths[global_idx] if global_idx < len(neural_lengths) else neural_len
                actual_label_len = min(actual_neural_len, label_len)

                seq_preds = preds[i, :actual_label_len].cpu().numpy()
                seq_labels = labels[i, :actual_label_len].cpu().numpy()

                non_zero_mask = seq_labels != 0
                if non_zero_mask.sum() > 0:
                    sample_predictions.extend(seq_preds[non_zero_mask])
                    sample_targets.extend(seq_labels[non_zero_mask])

            batch_count += 1

    sample_predictions = np.array(sample_predictions)
    sample_targets = np.array(sample_targets)

    print(f"Collected {len(sample_predictions)} samples for analysis")
    return sample_predictions, sample_targets

def plot_model_metrics(sample_predictions, sample_targets):
    if len(sample_predictions) > 0:
        correct_predictions = (sample_predictions == sample_targets)
        accuracy_per_class = {}
        for label in np.unique(sample_targets):
            mask = sample_targets == label
            if mask.sum() > 0:
                accuracy_per_class[label] = correct_predictions[mask].mean()

        plt.subplot(2, 3, 4)
        if accuracy_per_class:
            top_classes = dict(sorted(accuracy_per_class.items(), key=lambda x: x[1], reverse=True)[:20])
            plt.bar(top_classes.keys(), top_classes.values(), alpha=0.7, color='green')
            plt.xlabel('Class ID')
            plt.ylabel('Accuracy')
            plt.title('Top 20 Classes by Accuracy')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Accuracy per Class (No Data)')

        plt.subplot(2, 3, 5)
        top_n_classes = 10
        sample_label_counts = Counter(sample_targets)
        top_class_indices = np.array([label for label, _ in sample_label_counts.most_common(top_n_classes)])

        mask = np.isin(sample_targets, top_class_indices)
        if mask.sum() > 0:
            cm = confusion_matrix(sample_targets[mask], sample_predictions[mask], labels=top_class_indices)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=top_class_indices, yticklabels=top_class_indices)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix (Top {top_n_classes} Classes)')
        else:
            plt.text(0.5, 0.5, 'No data for confusion matrix', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Confusion Matrix (No Data)')
    else:
        for i in [4, 5]:
            plt.subplot(2, 3, i)
            plt.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Data Available')

def evaluate_confidence(model, train_loader, device):
    plt.subplot(2, 3, 6)
    try:
        with torch.no_grad():
            batch = next(iter(train_loader))
            neural_data, labels, padding_mask, _ = batch
            
            neural_data = neural_data.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(neural_data, src_key_padding_mask=padding_mask)
            probabilities = F.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probabilities, dim=-1)

            confidence_scores = []
            for i in range(neural_data.shape[0]):
                seq_len = min(neural_data.shape[1], labels.shape[1])
                seq_probs = max_probs[i, :seq_len].cpu().numpy()
                seq_labels = labels[i, :seq_len].cpu().numpy()

                non_zero_mask = seq_labels != 0
                confidence_scores.extend(seq_probs[non_zero_mask])

            if confidence_scores:
                plt.hist(confidence_scores, bins=30, alpha=0.7, edgecolor='black', color='purple')
                plt.xlabel('Maximum Probability')
                plt.ylabel('Frequency')
                plt.title('Distribution of Prediction Confidence')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Prediction Confidence (No Data)')
    except Exception as e:
        plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
        plt.title('Prediction Confidence (Error)')
        
    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/main_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_metrics(sample_predictions, sample_targets):
    print("\n=== DETAILED PERFORMANCE METRICS ===")
    if len(sample_predictions) > 0:
        overall_accuracy = (sample_predictions == sample_targets).mean()
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

        print("\nPer-class metrics for top 10 most frequent classes in sample:")
        sample_label_counts = Counter(sample_targets)
        top_10_classes = [label for label, _ in sample_label_counts.most_common(10)]

        for class_id in top_10_classes:
            class_mask = sample_targets == class_id
            if class_mask.sum() > 0:
                class_accuracy = (sample_predictions[class_mask] == class_id).mean()
                class_frequency = class_mask.mean()
                support = class_mask.sum()
                print(f"  Class {class_id:3d}: Accuracy={class_accuracy:.4f}, Frequency={class_frequency:.4f}, Support={support}")
    else:
        print("No prediction data available for detailed metrics")

def analyze_training_dynamics(model):
    print("\n=== TRAINING DYNAMICS ===")
    print("Model Weights Analysis:")
    weight_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() > 0:
            weight_stats.append({
                'Layer': name,
                'Mean': param.data.mean().item(),
                'Std': param.data.std().item(),
                'Min': param.data.min().item(),
                'Max': param.data.max().item(),
                'Parameters': param.numel()
            })

    if weight_stats:
        weight_df = pd.DataFrame(weight_stats)
        print(weight_df.to_string(index=False))
    else:
        print("No weight statistics available")

def plot_additional_visualizations(neural_lengths, label_lengths, label_counts):
    print("\n=== ADDITIONAL VISUALIZATIONS ===")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    if len(neural_lengths) == len(label_lengths):
        plt.scatter(neural_lengths, label_lengths, alpha=0.6)
        plt.xlabel('Neural Sequence Length')
        plt.ylabel('Label Sequence Length')
        plt.title('Neural vs Label Sequence Lengths')
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    if label_counts:
        top_30_labels = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:30])
        plt.bar(range(len(top_30_labels)), list(top_30_labels.values()), alpha=0.7)
        plt.xlabel('Class Rank')
        plt.ylabel('Frequency')
        plt.title('Class Frequency Distribution (Top 30)')
        plt.xticks(range(len(top_30_labels)), list(top_30_labels.keys()), rotation=45)

    plt.subplot(1, 3, 3)
    plt.plot(neural_lengths, marker='o', alpha=0.7, markersize=3)
    plt.xlabel('Trial Index')
    plt.ylabel('Sequence Length')
    plt.title('Sequence Length by Trial')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/additional_analysis_LSTM.png", dpi=300, bbox_inches='tight')
    plt.show()

def perform_data_quality_check(neural_lengths, all_labels, label_lengths):
    print("\n=== DATA QUALITY CHECK ===")
    print(f"Total trials: {len(neural_lengths)}")
    print(f"Total label tokens: {len(all_labels)}")
    if len(all_labels) > 0:
        print(f"Unique classes: {len(np.unique(all_labels))}")
        print(f"Label value range: {all_labels.min()} to {all_labels.max()}")

        zero_labels = (all_labels == 0).sum()
        print(f"Zero labels (potential padding): {zero_labels} ({zero_labels / len(all_labels):.2%})")

    if len(label_lengths) > 0 and np.mean(label_lengths) > 0:
        neural_to_label_ratio = neural_lengths.mean() / np.mean(label_lengths)
        print(f"Neural-to-label length ratio: {neural_to_label_ratio:.2f}")

def analyze_model_capacity(model, all_labels):
    print("\n=== MODEL CAPACITY ANALYSIS ===")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Parameters per class: {total_params / config.vocab_size:,.0f}")

    data_points = len(all_labels)
    print(f"Total data points: {data_points:,}")
    if data_points > 0:
        print(f"Parameters per data point: {total_params / data_points:.2f}")
    print("\n=== ANALYSIS COMPLETE ===")

def main():
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Enforce LSTM config for visualization_LSTM.py explicitly
    config.model_type = "LSTM"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Visualization LSTM] Using device: {device}")

    print("\n[Visualization LSTM] Discovering training files...")
    train_files = _discover_files(config.DATA_DIR, config.SESSION_GLOB, config.TRAIN_FILENAME)
    
    if not train_files:
        print(f"[ERROR] No train files found under {config.DATA_DIR}/{config.SESSION_GLOB}/{config.TRAIN_FILENAME}")
        return

    neural_lengths, label_sequences, all_labels, label_counts, label_lengths = analyze_data_distribution(train_files)

    print("\n[Visualization LSTM] Initializing DataLoader...")
    train_dataset = BrainDataset(train_files, is_test=False, max_len=config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print("\n[Visualization LSTM] Loading Model...")
    model = build_model(config, device)
    
    try:
        latest_ckpt = get_latest_checkpoint()
        print(f"[Visualization LSTM] Loading checkpoint from: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint: {e}")
        print("Using untrained model for visualization!")

    sample_predictions, sample_targets = evaluate_model_performance(model, train_loader, device, neural_lengths)
    
    plot_model_metrics(sample_predictions, sample_targets)
    evaluate_confidence(model, train_loader, device)
    
    print_detailed_metrics(sample_predictions, sample_targets)
    analyze_training_dynamics(model)
    plot_additional_visualizations(neural_lengths, label_lengths, label_counts)
    perform_data_quality_check(neural_lengths, all_labels, label_lengths)
    analyze_model_capacity(model, all_labels)

if __name__ == "__main__":
    main()