import os


class Config:
    # ─── Data Paths ──────────────────────────────────────────────────────────
    # Root directory that contains sub-folders like t15.2025.03.14/
    DATA_DIR: str = "/media/zzg/GJ_disk01/data/BCI/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final"
    # Glob pattern relative to DATA_DIR to discover session sub-folders
    SESSION_GLOB: str = "t15.*"
    TRAIN_FILENAME: str = "data_train.hdf5"
    VAL_FILENAME: str = "data_val.hdf5"
    TEST_FILENAME: str = "data_test.hdf5"

    # ─── Model ───────────────────────────────────────────────────────────────
    input_size: int = 512
    d_model: int = 768              #512,768,d_model 能被 nhead 整除
    nhead: int = 12                 #8,12
    num_layers: int = 12            #4,6,8,12
    dim_feedforward: int = 2048     #1024,2048
    dropout: float = 0.2

    # ─── Training ────────────────────────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 70
    max_seq_len: int = 500
    grad_clip: float = 1.0

    # ─── Data ────────────────────────────────────────────────────────────────
    vocab_size: int = 500  # Based on data exploration

    # ─── Checkpoint ──────────────────────────────────────────────────────────
    CHECKPOINT_DIR: str = "checkpoints"
    BEST_MODEL_NAME: str = "best_model.pth"

    # ─── MLFlow ──────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MLFLOW_EXPERIMENT_NAME: str = "brain-to-text-transformer"
    # Set to "" / None to skip registration
    MLFLOW_MODEL_NAME: str = "BrainTransformer"

    # ─── Submission ──────────────────────────────────────────────────────────
    SUBMISSION_FILE: str = "submission.csv"

    def __post_init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

    def as_dict(self) -> dict:
        """Return all config attributes as a flat dict (for MLFlow logging)."""
        return {
            k: v
            for k, v in self.__class__.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


# Singleton instance used throughout the project
config = Config()
