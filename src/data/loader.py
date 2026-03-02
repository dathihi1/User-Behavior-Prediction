"""Data loading utilities for variable-length sequences."""
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and manage competition data."""

    TARGET_COLS = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]

    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self._validate_data_dir()

    def _validate_data_dir(self) -> None:
        """Validate that data directory exists."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data (X_train, Y_train)."""
        x_train = self._load_sequences(self.data_dir / "X_train.csv")
        y_train = self._load_targets(self.data_dir / "Y_train.csv")
        logger.info(f"Loaded training data: X={len(x_train)}, Y={len(y_train)}")
        return x_train, y_train

    def load_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load validation data (X_val, Y_val)."""
        x_val = self._load_sequences(self.data_dir / "X_val.csv")
        y_val = self._load_targets(self.data_dir / "Y_val.csv")
        logger.info(f"Loaded validation data: X={len(x_val)}, Y={len(y_val)}")
        return x_val, y_val

    def load_test(self) -> pd.DataFrame:
        """Load test data (X_test)."""
        x_test = self._load_sequences(self.data_dir / "X_test.csv")
        logger.info(f"Loaded test data: X={len(x_test)}")
        return x_test

    def load_all(self) -> dict:
        """Load all datasets."""
        x_train, y_train = self.load_train()
        x_val, y_val = self.load_val()
        x_test = self.load_test()

        return {
            "X_train": x_train,
            "Y_train": y_train,
            "X_val": x_val,
            "Y_val": y_val,
            "X_test": x_test,
        }

    def _load_sequences(self, path: Path) -> pd.DataFrame:
        """Load sequence data from CSV."""
        df = pd.read_csv(path)
        logger.info(f"Loaded {path.name}: {df.shape}")
        return df

    def _load_targets(self, path: Path) -> pd.DataFrame:
        """Load target data from CSV."""
        df = pd.read_csv(path)
        logger.info(f"Loaded {path.name}: {df.shape}")
        return df

    @staticmethod
    def parse_sequence(row: pd.Series, exclude_cols: List[str] = None) -> np.ndarray:
        """Parse a row into sequence of actions, excluding specified columns."""
        if exclude_cols is None:
            exclude_cols = ["id"]

        # Get all values except excluded columns
        seq_cols = [c for c in row.index if c not in exclude_cols]
        values = row[seq_cols].values

        # Remove NaN values (variable length sequences may have trailing NaN)
        valid_values = values[~pd.isna(values)]
        return valid_values.astype(np.int64)

    @staticmethod
    def get_sequence_lengths(df: pd.DataFrame, exclude_cols: List[str] = None) -> np.ndarray:
        """Get sequence lengths for all rows."""
        if exclude_cols is None:
            exclude_cols = ["id"]

        seq_cols = [c for c in df.columns if c not in exclude_cols]
        # Count non-NaN values per row
        lengths = df[seq_cols].notna().sum(axis=1).values
        return lengths

    def merge_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge train and validation sets for final training (allowed per rules)."""
        x_train, y_train = self.load_train()
        x_val, y_val = self.load_val()

        x_merged = pd.concat([x_train, x_val], ignore_index=True)
        y_merged = pd.concat([y_train, y_val], ignore_index=True)

        logger.info(f"Merged train+val: X={len(x_merged)}, Y={len(y_merged)}")
        return x_merged, y_merged
