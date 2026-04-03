# --- Visualization and Analysis Cell ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import Counter
import numpy as np

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Data Distribution Analysis
print("=== DATA DISTRIBUTION ANALYSIS ===")

# Analyze neural sequence lengths
neural_lengths = []
label_sequences = []

with h5py.File(TRAIN_FILE, 'r') as f:
    for trial_key in f.keys():
        trial = f[trial_key]
        neural_lengths.append(trial['input_features'].shape[0])
        label_sequences.append(trial['seq_class_ids'][:])

neural_lengths = np.array(neural_lengths)
print(f"Neural sequence length statistics:")
print(f"  Min: {neural_lengths.min()}")
print(f"  Max: {neural_lengths.max()}")
print(f"  Mean: {neural_lengths.mean():.1f}")
print(f"  Std: {neural_lengths.std():.1f}")

# Plot neural sequence length distribution
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.hist(neural_lengths, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Neural Sequence Lengths')
plt.grid(True, alpha=0.3)

# 2. Label Distribution Analysis
all_labels = np.concatenate(label_sequences)
label_counts = Counter(all_labels)

plt.subplot(2, 3, 2)
top_labels = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20])
plt.bar(top_labels.keys(), top_labels.values(), alpha=0.7)
plt.xlabel('Label ID')
plt.ylabel('Frequency')
plt.title('Top 20 Most Frequent Labels')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 3. Label Sequence Length Analysis
label_lengths = [len(seq) for seq in label_sequences]
plt.subplot(2, 3, 3)
plt.hist(label_lengths, bins=20, alpha=0.7, edgecolor='black', color='orange')
plt.xlabel('Label Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Label Sequence Lengths')
plt.grid(True, alpha=0.3)

# 4. Model Performance Analysis
print("\n=== MODEL PERFORMANCE ANALYSIS ===")

# Get sample predictions for analysis - FIXED VERSION
model.eval()
sample_predictions = []
sample_targets = []

with torch.no_grad():
    batch_count = 0
    for neural_data, labels, padding_mask, _ in train_loader:
        if batch_count >= 3:  # Use first 3 batches for analysis
            break

        neural_data = neural_data.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)

        logits = model(neural_data, src_key_padding_mask=padding_mask)

        batch_size, neural_len, vocab_size = logits.shape
        label_len = labels.shape[1]

        # Align logits with labels - take first label_len time steps
        if neural_len >= label_len:
            logits_aligned = logits[:, :label_len, :]
        else:
            # If neural sequence is shorter, we can't make all predictions
            logits_aligned = logits

        preds = torch.argmax(logits_aligned, dim=-1)

        # Process each sequence in the batch individually
        for i in range(batch_size):
            # Get the actual sequence length for this sample
            actual_neural_len = neural_lengths[batch_count * batch_size + i] if (batch_count * batch_size + i) < len(
                neural_lengths) else neural_len
            actual_label_len = min(actual_neural_len, label_len)

            # Take predictions for the actual sequence length
            seq_preds = preds[i, :actual_label_len].cpu().numpy()
            seq_labels = labels[i, :actual_label_len].cpu().numpy()

            # Only add non-zero labels (assuming 0 is padding)
            non_zero_mask = seq_labels != 0
            if non_zero_mask.sum() > 0:
                sample_predictions.extend(seq_preds[non_zero_mask])
                sample_targets.extend(seq_labels[non_zero_mask])

        batch_count += 1

sample_predictions = np.array(sample_predictions)
sample_targets = np.array(sample_targets)

print(f"Collected {len(sample_predictions)} samples for analysis")

if len(sample_predictions) > 0:
    # Calculate accuracy per class
    correct_predictions = (sample_predictions == sample_targets)
    accuracy_per_class = {}
    for label in np.unique(sample_targets):
        mask = sample_targets == label
        if mask.sum() > 0:
            accuracy_per_class[label] = correct_predictions[mask].mean()

    # Plot accuracy per class
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

    # 5. Confusion Matrix (for top classes)
    plt.subplot(2, 3, 5)
    if len(sample_targets) > 0:
        top_n_classes = 10
        # Get most frequent classes in our sample
        sample_label_counts = Counter(sample_targets)
        top_class_indices = np.array([label for label, _ in sample_label_counts.most_common(top_n_classes)])

        # Filter predictions for top classes
        mask = np.isin(sample_targets, top_class_indices)
        if mask.sum() > 0:
            cm = confusion_matrix(sample_targets[mask], sample_predictions[mask], labels=top_class_indices)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=top_class_indices, yticklabels=top_class_indices)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix (Top {top_n_classes} Classes)')
        else:
            plt.text(0.5, 0.5, 'No data for confusion matrix', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Confusion Matrix (No Data)')
    else:
        plt.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Confusion Matrix (No Data)')

else:
    # Placeholder plots if no prediction data
    for i in [4, 5]:
        plt.subplot(2, 3, i)
        plt.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Data Available')

# 6. Prediction Confidence Analysis
plt.subplot(2, 3, 6)
try:
    with torch.no_grad():
        # Get confidence scores for a single batch
        neural_data, labels, padding_mask, _ = next(iter(train_loader))
        neural_data = neural_data.to(device)
        padding_mask = padding_mask.to(device)

        logits = model(neural_data, src_key_padding_mask=padding_mask)
        probabilities = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probabilities, dim=-1)

        # Process each sequence individually
        confidence_scores = []
        for i in range(neural_data.shape[0]):
            seq_len = min(neural_data.shape[1], labels.shape[1])
            seq_probs = max_probs[i, :seq_len].cpu().numpy()
            seq_labels = labels[i, :seq_len].cpu().numpy()

            # Only keep non-padded positions
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
plt.show()

# 7. Detailed Performance Metrics
print("\n=== DETAILED PERFORMANCE METRICS ===")
if len(sample_predictions) > 0:
    overall_accuracy = (sample_predictions == sample_targets).mean()
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Per-class metrics for top classes
    print("\nPer-class metrics for top 10 most frequent classes in sample:")
    sample_label_counts = Counter(sample_targets)
    top_10_classes = [label for label, _ in sample_label_counts.most_common(10)]

    for class_id in top_10_classes:
        class_mask = sample_targets == class_id
        if class_mask.sum() > 0:
            class_accuracy = (sample_predictions[class_mask] == class_id).mean()
            class_frequency = class_mask.mean()
            support = class_mask.sum()
            print(
                f"  Class {class_id:3d}: Accuracy={class_accuracy:.4f}, Frequency={class_frequency:.4f}, Support={support}")
else:
    print("No prediction data available for detailed metrics")

# 8. Training Dynamics Analysis
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
    # Create a summary DataFrame
    weight_df = pd.DataFrame(weight_stats)
    print(weight_df.to_string(index=False))
else:
    print("No weight statistics available")

# 9. Additional Visualizations
plt.figure(figsize=(15, 5))

# Neural vs Label length correlation
plt.subplot(1, 3, 1)
plt.scatter(neural_lengths, label_lengths, alpha=0.6)
plt.xlabel('Neural Sequence Length')
plt.ylabel('Label Sequence Length')
plt.title('Neural vs Label Sequence Lengths')
plt.grid(True, alpha=0.3)

# Class distribution (full dataset)
plt.subplot(1, 3, 2)
top_30_labels = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:30])
plt.bar(range(len(top_30_labels)), list(top_30_labels.values()), alpha=0.7)
plt.xlabel('Class Rank')
plt.ylabel('Frequency')
plt.title('Class Frequency Distribution (Top 30)')
plt.xticks(range(len(top_30_labels)), list(top_30_labels.keys()), rotation=45)

# Sequence length over trials
plt.subplot(1, 3, 3)
plt.plot(neural_lengths, marker='o', alpha=0.7, markersize=3)
plt.xlabel('Trial Index')
plt.ylabel('Sequence Length')
plt.title('Sequence Length by Trial')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. Data Quality Check
print("\n=== DATA QUALITY CHECK ===")
print(f"Total trials: {len(neural_lengths)}")
print(f"Total label tokens: {len(all_labels)}")
print(f"Unique classes: {len(np.unique(all_labels))}")
print(f"Label value range: {all_labels.min()} to {all_labels.max()}")

# Check for any anomalies
zero_labels = (all_labels == 0).sum()
print(f"Zero labels (potential padding): {zero_labels} ({zero_labels / len(all_labels):.2%})")

# Check sequence length consistency
neural_to_label_ratio = neural_lengths.mean() / np.mean(label_lengths)
print(f"Neural-to-label length ratio: {neural_to_label_ratio:.2f}")

# 11. Model Capacity Analysis
print("\n=== MODEL CAPACITY ANALYSIS ===")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
print(f"Parameters per class: {total_params / config.vocab_size:,.0f}")

# Estimate model capacity vs data size
data_points = len(all_labels)
print(f"Total data points: {data_points:,}")
print(f"Parameters per data point: {total_params / data_points:.2f}")

print("\n=== ANALYSIS COMPLETE ===")