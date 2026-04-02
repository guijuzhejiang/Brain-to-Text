"""
dataset.py – Brain-to-Text Dataset

Supports loading from **multiple** HDF5 files (one per recording session).
"""

from __future__ import annotations

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

from config import config

class BrainDataset(Dataset):
    """
    Load brain-signal trials from one **or more** HDF5 files.

    Parameters
    ----------
    hdf5_files : list[str]
        Paths to HDF5 files.  Each file may contain multiple trials stored as
        top-level groups.
    is_test : bool
        When True, dummy labels are returned (no 'seq_class_ids' expected).
    max_len : int
        Maximum label sequence length (pad / truncate).
    """

    def __init__(
            self,
            hdf5_files: List[str],
            is_test: bool = False,
            max_len: int = config.max_seq_len,
    ) -> None:
        super().__init__()

        if isinstance(hdf5_files, str):
            hdf5_files = [hdf5_files]

        self.is_test = is_test
        self.max_len = max_len

        # Build a flat index: (file_path, trial_key)
        self._index: List[Tuple[str, str]] = []
        for path in hdf5_files:
            with h5py.File(path, "r") as f:
                for key in f.keys():
                    self._index.append((path, key))

        print(f"[BrainDataset] Loaded {len(self._index)} trials "
              f"from {len(hdf5_files)} file(s).")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        file_path, trial_key = self._index[idx]

        with h5py.File(file_path, "r") as f:
            trial = f[trial_key]
            neural_data = torch.tensor(
                trial["input_features"][:], dtype=torch.float32
            )

            if not self.is_test:
                labels = torch.tensor(
                    trial["seq_class_ids"][:], dtype=torch.long
                )
                labels = torch.clamp(labels, 0, config.vocab_size - 1)
                # Truncate / pad to max_len
                if len(labels) > self.max_len:
                    labels = labels[: self.max_len]
                elif len(labels) < self.max_len:
                    labels = F.pad(
                        labels, (0, self.max_len - len(labels)), value=0
                    )
            else:
                labels = torch.zeros(self.max_len, dtype=torch.long)

        # Use "file_stem/trial_key" as a unique id
        unique_id = f"{file_path.split('/')[-2]}/{trial_key}"
        return neural_data, labels, unique_id

def collate_fn(batch):
    """
    Pad neural sequences to the longest in the batch and build a
    boolean padding mask (True = pad position, ignored in attention).
    """
    neural_data = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    trial_keys = [item[2] for item in batch]

    neural_padded = pad_sequence(neural_data, batch_first=True)
    neural_lengths = torch.tensor(
        [len(x) for x in neural_data], dtype=torch.long
    )
    max_len = neural_padded.shape[1]
    padding_mask = (
            torch.arange(max_len).expand(len(neural_lengths), max_len)
            >= neural_lengths.unsqueeze(1)
    )

    return neural_padded, labels, padding_mask, trial_keys


def explore_files(hdf5_files: List[str], n_preview: int = 5) -> None:
    """Print a brief summary of the supplied HDF5 files."""
    total_trials = 0
    for path in hdf5_files:
        with h5py.File(path, "r") as f:
            trials = list(f.keys())
            total_trials += len(trials)
            print(f"\n[explore] {path}")
            print(f"  Trials: {len(trials)}")
            for key in trials[:n_preview]:
                neural_len = f[key]["input_features"].shape[0]
                label_len = (
                    f[key]["seq_class_ids"].shape[0]
                    if "seq_class_ids" in f[key]
                    else "N/A"
                )
                print(f"  {key}: neural={neural_len}, labels={label_len}")
    print(f"\n[explore] Grand total: {total_trials} trials")
