"""
eda.py - Exploratory Data Analysis for Brain-to-Text Competition Data
Usage:
    python eda.py              # Run all analyses
    python eda.py --section 1  # Run a specific section (1-5)

Sections:
    1 - Load all metadata (train + val + test) → dataset_overview.csv
    2 - Memory / NaN / duplicate diagnostics
    3 - Sentence-length, duration, WPM distributions
    4 - Corpus distribution & per-corpus statistics
    5 - Channel correlation heatmaps
"""

import argparse
import os
import string
import traceback
from datetime import datetime
from glob import glob

import h5py
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – saves to file, no popup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from config import config

# ─── Plot style ───────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")
pd.set_option("display.max_colwidth", 100)

# ─── Constants ────────────────────────────────────────────────────────────────
NEURAL_DATA_KEY   = "input_features"
TRANSCRIPTION_KEY = "transcription"

# Neural channel → brain region mapping (512 channels total)
FEATURE_MAP = {
    "ventral_6v_thresh": list(range(0,   64)),
    "area_4_thresh":     list(range(64,  128)),
    "area_55b_thresh":   list(range(128, 192)),
    "dorsal_6v_thresh":  list(range(192, 256)),
    "ventral_6v_sbp":    list(range(256, 320)),
    "area_4_sbp":        list(range(320, 384)),
    "area_55b_sbp":      list(range(384, 448)),
    "dorsal_6v_sbp":     list(range(448, 512)),
}

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def _save_fig(name: str) -> None:
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


# ─── Data loading helpers ─────────────────────────────────────────────────────

def _parse_date(session: str) -> str:
    """Convert 't15.2023.08.11' → '2023-08-11'."""
    try:
        date_part = session.split(".", 1)[1]          # '2023.08.11'
        dt = datetime.strptime(date_part, "%Y.%m.%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def load_metadata_from_hdf5(file_path: str) -> list[dict]:
    """
    Load per-trial metadata from a train/val HDF5 file.

    HDF5 layout (per trial group):
        Datasets : input_features  (T, 512)
                   seq_class_ids   (T,)       – integer phoneme labels
                   transcription   (T,)       – ASCII codes
        Attrs    : block_num, n_time_steps, sentence_label, seq_len,
                   session, trial_num
    """
    metadata = []
    try:
        with h5py.File(file_path, "r") as f:
            for trial in f.keys():
                grp = f[trial]
                if not isinstance(grp, h5py.Group):
                    continue
                keys = list(grp.keys())
                if NEURAL_DATA_KEY not in keys or TRANSCRIPTION_KEY not in keys:
                    continue

                block_num      = grp.attrs["block_num"]
                sentence_label = grp.attrs["sentence_label"]
                phoneme_seq_len = grp.attrs["seq_len"]
                session        = grp.attrs["session"]
                trial_num      = grp.attrs["trial_num"]

                input_features  = grp[NEURAL_DATA_KEY][()]          # (T, 512)
                seq_class_ids   = grp["seq_class_ids"][()]
                seq_transcription = list(grp["transcription"][()])

                num_time_bins, num_channels = input_features.shape
                num_words = len(sentence_label.split())

                metadata.append({
                    "session":                   session,
                    "trial_id":                  trial,
                    "block_number":              block_num,
                    "trial_num":                 trial_num,
                    "corpus":                    None,        # filled by section 4 if needed
                    "num_time_bins":             num_time_bins,
                    "num_channels":              num_channels,
                    "neural_features":           input_features,
                    "phoneme_labels":            seq_class_ids,
                    "num_of_phoneme_labels":     int(phoneme_seq_len),
                    "transcription_ASCII":       seq_transcription,
                    "transcription_text":        sentence_label,
                    "num_texts":                 num_words,
                })
    except Exception as e:
        print(f"  [ERROR] {file_path}: {e}")
        traceback.print_exc()
    return metadata


def load_test_metadata_from_hdf5(file_path: str) -> list[dict]:
    """
    Load per-trial metadata from a test HDF5 file.
    Test trials only carry input_features (no labels).
    """
    metadata = []
    try:
        with h5py.File(file_path, "r") as f:
            for trial in f.keys():
                grp = f[trial]
                if not isinstance(grp, h5py.Group):
                    continue
                if NEURAL_DATA_KEY not in grp:
                    continue

                block_num  = grp.attrs["block_num"]
                session    = grp.attrs["session"]
                trial_num  = grp.attrs["trial_num"]
                input_features = grp[NEURAL_DATA_KEY][()]
                num_time_bins, num_channels = input_features.shape

                metadata.append({
                    "session":          session,
                    "trial_id":         trial,
                    "block_number":     block_num,
                    "trial_num":        trial_num,
                    "corpus":           None,
                    "num_time_bins":    num_time_bins,
                    "num_channels":     num_channels,
                    "neural_features":  input_features,
                    "phoneme_labels":   None,
                    "num_of_phoneme_labels": None,
                    "transcription_ASCII": None,
                    "transcription_text":  None,
                    "num_texts":           None,
                })
    except Exception as e:
        print(f"  [ERROR] {file_path}: {e}")
        traceback.print_exc()
    return metadata


def load_all_metadata(force_reload: bool = False) -> pd.DataFrame:
    """
    Scan DATA_DIR for all session folders, load train/val/test HDF5 files,
    and return a consolidated DataFrame.
    """
    base = config.DATA_DIR
    session_dirs = sorted(
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    )
    print(f"  Found {len(session_dirs)} session directories under {base}")

    all_metadata = []
    for session in tqdm(session_dirs, desc="Sessions"):
        session_path = os.path.join(base, session)
        for split in ("train", "val", "test"):
            fp = os.path.join(session_path, f"data_{split}.hdf5")
            if not os.path.exists(fp):
                continue
            if split == "test":
                records = load_test_metadata_from_hdf5(fp)
            else:
                records = load_metadata_from_hdf5(fp)
            for r in records:
                r["split"] = split
            all_metadata.extend(records)

    df = pd.DataFrame(all_metadata)
    return df


# ─── Sections ─────────────────────────────────────────────────────────────────

def section1_dataset_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION 1 — Dataset Overview")
    print("=" * 70)
    print(df.dtypes)
    print(f"\nTotal trials: {len(df)}")
    print("\nTrials per split:")
    print(df["split"].value_counts().to_string())

    # Per-session summary
    summary = (
        df.groupby(["session", "split"])
        .agg(trials=("trial_id", "count"),
             channels=("num_channels", "first"))
        .reset_index()
    )
    print("\nPer-session summary (train/val only):")
    print(summary[summary["split"] != "test"].to_string(index=False))

    csv_path = os.path.join(PLOT_DIR, "dataset_overview.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n  → Overview CSV: {csv_path}")


def section2_diagnostics(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION 2 — Memory / NaN / Duplicate Diagnostics")
    print("=" * 70)

    def fmt(b):
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.2f} {unit}"
            b /= 1024
        return f"{b:.2f} TB"

    shallow = df.memory_usage(index=True, deep=False).sum()
    deep    = df.memory_usage(index=True, deep=True).sum()
    print(f"Shallow memory: {fmt(shallow)}")
    print(f"Deep memory   : {fmt(deep)}")

    print("\nNaN counts per column:")
    print(df.isna().sum().to_string())

    dup = df.duplicated(subset=["session", "trial_id"]).sum()
    print(f"\nDuplicate (session, trial_id) entries: {dup}")


def section3_distributions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION 3 — Sentence Length / Duration / WPM Distributions")
    print("=" * 70)

    # Duration
    df = df.copy()
    df["duration_s"] = df["num_time_bins"] * 0.02
    df["wpm"] = df["num_texts"] / df["duration_s"] * 60

    # ── Sentence-word-count distribution ──
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=df.dropna(subset=["num_texts"]), x="num_texts",
                 bins=np.arange(0, 40, 1), kde=False, ax=ax)
    ax.set_title("Distribution of Sentence Length (Word Count)")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Count")
    _save_fig("sentence_length_distribution.png")
    print("Sentence length stats:")
    print(df["num_texts"].describe().to_string())

    # ── Duration + WPM ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="duration_s", bins=50, ax=axes[0])
    axes[0].set_title("Trial Duration")
    axes[0].set_xlabel("Duration (s)")

    sns.histplot(data=df.dropna(subset=["wpm"]), x="wpm", bins=50, ax=axes[1])
    axes[1].set_title("Speaking Rate (WPM)")
    axes[1].set_xlabel("Words Per Minute")
    plt.tight_layout()
    _save_fig("duration_wpm_distribution.png")

    # ── Time bins vs Phoneme labels scatter ──
    subset = df[df["split"].isin(["train", "val"])].dropna(
        subset=["num_of_phoneme_labels"]
    ).copy()
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(subset["num_time_bins"], subset["num_of_phoneme_labels"],
                   alpha=0.3, s=10)
        ax.set_title("Neural Time Bins vs. Phoneme Labels")
        ax.set_xlabel("Neural Time Bins")
        ax.set_ylabel("Number of Phoneme Labels")
        corr = subset[["num_time_bins", "num_of_phoneme_labels"]].corr().iloc[0, 1]
        ax.set_title(f"Time Bins vs Phoneme Labels  (r = {corr:.2f})")
        _save_fig("timebins_vs_phonemes_scatter.png")

        # Ratio analysis
        subset["ratio"] = subset["num_time_bins"] / subset["num_of_phoneme_labels"]
        ratio = subset["ratio"]
        print(f"\nTime-bins / Phoneme-labels ratio — "
              f"mean={ratio.mean():.2f}, median={ratio.median():.2f}, "
              f"std={ratio.std():.2f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(ratio, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0].axvline(ratio.mean(),   color="red",   linestyle="--",
                        label=f"Mean {ratio.mean():.2f}")
        axes[0].axvline(ratio.median(), color="green", linestyle="--",
                        label=f"Median {ratio.median():.2f}")
        axes[0].set_title("Ratio Distribution")
        axes[0].legend()

        axes[1].boxplot(ratio)
        axes[1].set_title("Box Plot")

        stats.probplot(ratio, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot")

        plt.tight_layout()
        _save_fig("ratio_distribution.png")

    # ── Vocabulary (word-level) ──
    train_texts = df[df["split"] != "test"]["transcription_text"].dropna()
    if not train_texts.empty:
        all_words = (
            " ".join(train_texts)
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
            .split()
        )
        vocab = set(all_words)
        print(f"\nWord-level vocabulary size : {len(vocab)}")
        print(f"Total tokens               : {len(all_words)}")


def section4_corpus_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION 4 — Corpus / Sentence Distribution")
    print("=" * 70)

    # Derive corpus from session date + block_number if a description CSV exists;
    # otherwise fall back to showing 'corpus' col (may be None).
    if df["corpus"].isna().all():
        print("  [INFO] No corpus mapping available. "
              "Showing trial counts per (session, split) instead.")
    
    print("\nTrial counts per split:")
    print(df["split"].value_counts().to_string())

    # Per-split corpus bar chart (if corpus present)
    if not df["corpus"].isna().all():
        splits = [s for s in ("train", "val", "test") if s in df["split"].values]
        fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 6),
                                 sharey=True)
        if len(splits) == 1:
            axes = [axes]
        for ax, split in zip(axes, splits):
            sub = df[df["split"] == split]
            order = sub["corpus"].value_counts().index
            total = len(sub)
            sns.countplot(y="corpus", data=sub, order=order,
                          palette="pastel", ax=ax)
            ax.set_title(f"{split.capitalize()} (n={total})")
            for p in ax.patches:
                count = p.get_width()
                ax.text(count + total * 0.005,
                        p.get_y() + p.get_height() / 2,
                        f"{100*count/total:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        _save_fig("corpus_distribution.png")

        # Per-corpus averages
        df2 = df.copy()
        df2["duration_s"] = df2["num_time_bins"] * 0.02
        df2["wpm"] = df2["num_texts"] / df2["duration_s"] * 60
        corpus_summary = (
            df2.groupby("corpus")
            .agg(mean_time_bins=("num_time_bins", "mean"),
                 mean_phoneme=("num_of_phoneme_labels", "mean"),
                 mean_duration_s=("duration_s", "mean"),
                 mean_wpm=("wpm", "mean"),
                 trial_count=("trial_id", "count"))
            .reset_index()
        )
        print("\nPer-corpus summary:")
        print(corpus_summary.to_string(index=False))

        order = corpus_summary.sort_values("trial_count", ascending=False)["corpus"]
        metrics = [
            ("mean_time_bins",  "Avg Time Bins"),
            ("mean_phoneme",    "Avg Phoneme Count"),
            ("mean_duration_s", "Avg Duration (s)"),
            ("mean_wpm",        "Avg WPM"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (col, title) in zip(axes.flatten(), metrics):
            sns.barplot(x="corpus", y=col, data=corpus_summary,
                        order=order, palette="viridis", ax=ax)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        _save_fig("corpus_per_metric.png")
    else:
        # Simple bar: trials per session
        sub_train = df[df["split"] == "train"]
        counts = sub_train["session"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(max(12, len(counts)), 5))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Train Trials per Session")
        ax.set_xlabel("Session")
        ax.set_ylabel("Trials")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        _save_fig("train_trials_per_session.png")


def section5_channel_correlations(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION 5 — Channel Correlation Heatmaps")
    print("=" * 70)

    train_val = df[df["split"] != "test"].copy()
    if train_val.empty or "neural_features" not in train_val.columns:
        print("  No train/val neural features available.")
        return

    neural_list = [
        nf for nf in train_val["neural_features"] if nf is not None
    ]
    if not neural_list:
        print("  neural_features column is empty.")
        return

    print(f"  Computing channel means over {len(neural_list)} trials …")
    # Average over time axis → (n_trials, 512)
    channel_means = np.array([trial.mean(axis=0) for trial in neural_list])
    num_channels = channel_means.shape[1]
    print(f"  channel_means shape: {channel_means.shape}")

    # Full correlation matrix
    corr_full = np.corrcoef(channel_means.T)   # (512, 512)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_full, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_step = max(1, num_channels // 16)
    ticks = np.arange(0, num_channels, tick_step)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, rotation=45, fontsize=7)
    ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Channel Index")
    ax.set_title("Full Channel Correlation Matrix (all 512 channels)")
    _save_fig("channel_corr_full.png")

    # Spike-band-power (last 256 channels) correlation
    if num_channels >= 512:
        sbp = channel_means[:, 256:]
        corr_sbp = np.corrcoef(sbp.T)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr_sbp, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(0, 256, 16)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, fontsize=7)
        ax.set_yticklabels(ticks, fontsize=7)
        ax.set_xlabel("Channel Index (SBP, ch 256-511)")
        ax.set_ylabel("Channel Index (SBP, ch 256-511)")
        ax.set_title("Spike-Band-Power Channel Correlation")
        _save_fig("channel_corr_sbp.png")


def section5b_single_trial_heatmap(df: pd.DataFrame) -> None:
    """Visualise one trial's raw neural activity (all 512 channels × time)."""
    subset = df[df["split"] != "test"].dropna(subset=["transcription_text", "neural_features"])
    if subset.empty:
        return
    row = subset.iloc[0]
    neural = row["neural_features"]      # (T, 512)
    sentence = row["transcription_text"]
    time_s = np.arange(neural.shape[0]) * 0.02

    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(
        neural.T,
        aspect=0.02,
        cmap="magma",
        interpolation="none",
        vmin=0, vmax=8,
        extent=[time_s[0], time_s[-1], neural.shape[1], 0],
    )
    for boundary in [64, 128, 192, 256, 320, 384, 448]:
        ax.axhline(boundary - 0.5, color="white", linestyle="--", linewidth=0.8)
    plt.colorbar(im, ax=ax, label="Neural Activity")
    ax.set_yticks([32, 96, 160, 224, 288, 352, 416, 480])
    ax.set_yticklabels([
        "v6v (Thresh)", "Area4 (Thresh)", "55b (Thresh)", "d6v (Thresh)",
        "v6v (SBP)",    "Area4 (SBP)",   "55b (SBP)",   "d6v (SBP)",
    ], fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_title(f'Single-Trial Neural Activity\n"{sentence}"', fontsize=16)
    plt.tight_layout()
    _save_fig("single_trial_heatmap.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Brain-to-Text EDA")
    parser.add_argument("--section", type=int, choices=[1, 2, 3, 4, 5],
                        default=None,
                        help="Run one section only (1-5). Omit for all.")
    parser.add_argument("--reload", action="store_true",
                        help="Force re-scan of HDF5 files (ignore cache).")
    args = parser.parse_args()

    print(f"\nData dir : {config.DATA_DIR}")
    print(f"Plots    : {PLOT_DIR}\n")

    df = load_all_metadata(force_reload=args.reload)
    print(f"Loaded DataFrame: {len(df)} rows, {len(df.columns)} columns\n")

    sections = {
        1: lambda: section1_dataset_overview(df),
        2: lambda: section2_diagnostics(df),
        3: lambda: section3_distributions(df),
        4: lambda: section4_corpus_distribution(df),
        5: lambda: (section5_channel_correlations(df),
                    section5b_single_trial_heatmap(df)),
    }

    if args.section is not None:
        sections[args.section]()
    else:
        for fn in sections.values():
            fn()

    print(f"\n✅  EDA complete. All plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
