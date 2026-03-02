"""Sequence preprocessing utilities."""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import get_logger

logger = get_logger(__name__)


class SequencePreprocessor:
    """Preprocess variable-length sequences for model input.

    Uses **dense vocab remapping**: original sparse action IDs (e.g. 102–24438)
    are remapped to contiguous indices 1..N so the embedding table has exactly
    N+1 rows (0 = padding) instead of max_id+1 rows. This can reduce embedding
    size by 96x on this dataset and ensures every embedding slot gets gradient.
    """

    def __init__(
        self,
        max_length: int = 512,
        padding_strategy: str = "post",
        truncation_strategy: str = "pre",
        padding_value: int = 0,
    ):
        self.max_length = max_length
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.padding_value = padding_value
        # vocab_mapping: raw_id -> dense_index (1-based; 0 reserved for padding)
        self.vocab_mapping: dict = {}
        self.vocab_size: int = 0   # = len(unique_tokens) + 1 (for padding)
        self._fitted = False

    def fit(self, sequences: List[np.ndarray]) -> "SequencePreprocessor":
        """Fit preprocessor: build dense vocab mapping from training sequences."""
        all_tokens = np.concatenate(sequences)
        unique_tokens = sorted(set(all_tokens.tolist()))
        # Assign dense indices starting at 1; 0 is reserved for padding
        self.vocab_mapping = {tok: idx + 1 for idx, tok in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens) + 1   # +1 for padding token 0
        self._fitted = True
        logger.info(
            f"Fitted preprocessor: {len(unique_tokens)} unique tokens → "
            f"dense vocab_size={self.vocab_size} "
            f"(was {int(all_tokens.max()) + 1} with raw IDs)"
        )
        return self

    def _remap(self, seq: np.ndarray) -> np.ndarray:
        """Map raw action IDs to dense indices. Unknown tokens → 0 (padding)."""
        return np.array(
            [self.vocab_mapping.get(int(t), 0) for t in seq], dtype=np.int64
        )

    def transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Remap to dense indices then pad/truncate to fixed length."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        padded = []
        for seq in sequences:
            remapped = self._remap(seq)
            padded_seq = self._pad_sequence(remapped)
            padded.append(padded_seq)

        return np.array(padded)

    def fit_transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Fit and transform sequences."""
        self.fit(sequences)
        return self.transform(sequences)

    def _pad_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Pad or truncate a single sequence to max_length."""
        seq_len = len(seq)

        if seq_len > self.max_length:
            # Truncate
            if self.truncation_strategy == "pre":
                # Keep most recent actions
                seq = seq[-self.max_length:]
            else:
                # Keep earliest actions
                seq = seq[:self.max_length]
        elif seq_len < self.max_length:
            # Pad
            pad_length = self.max_length - seq_len
            padding = np.full(pad_length, self.padding_value)

            if self.padding_strategy == "post":
                seq = np.concatenate([seq, padding])
            else:
                seq = np.concatenate([padding, seq])

        return seq.astype(np.int64)

    def get_attention_mask(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Generate attention mask (1 for real tokens, 0 for padding)."""
        masks = []
        for seq in sequences:
            seq_len = min(len(seq), self.max_length)

            if self.padding_strategy == "post":
                mask = np.concatenate([
                    np.ones(seq_len),
                    np.zeros(self.max_length - seq_len)
                ])
            else:
                mask = np.concatenate([
                    np.zeros(self.max_length - seq_len),
                    np.ones(seq_len)
                ])

            masks.append(mask)

        return np.array(masks, dtype=np.float32)


class TargetEncoder:
    """Encode multi-output targets."""

    def __init__(self, target_cols: List[str]):
        self.target_cols = target_cols
        self.encoders = {col: LabelEncoder() for col in target_cols}
        self.num_classes = {}
        self._fitted = False

    def fit(self, y: pd.DataFrame) -> "TargetEncoder":
        """Fit encoders for each target column."""
        for col in self.target_cols:
            self.encoders[col].fit(y[col])
            self.num_classes[col] = len(self.encoders[col].classes_)

        self._fitted = True
        logger.info(f"Fitted target encoder: {self.num_classes}")
        return self

    def transform(self, y: pd.DataFrame) -> np.ndarray:
        """Transform targets to encoded array."""
        if not self._fitted:
            raise RuntimeError("Encoder must be fitted before transform")

        encoded = np.zeros((len(y), len(self.target_cols)), dtype=np.int64)
        for i, col in enumerate(self.target_cols):
            encoded[:, i] = self.encoders[col].transform(y[col])

        return encoded

    def fit_transform(self, y: pd.DataFrame) -> np.ndarray:
        """Fit and transform targets."""
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_encoded: np.ndarray) -> pd.DataFrame:
        """Inverse transform encoded targets to original values."""
        if not self._fitted:
            raise RuntimeError("Encoder must be fitted before inverse_transform")

        decoded = {}
        for i, col in enumerate(self.target_cols):
            decoded[col] = self.encoders[col].inverse_transform(
                y_encoded[:, i].astype(int)
            )

        return pd.DataFrame(decoded)


def extract_sequences_from_df(
    df: pd.DataFrame, id_col: str = "id"
) -> Tuple[List[str], List[np.ndarray]]:
    """Extract sequences and IDs from DataFrame."""
    ids = df[id_col].tolist() if id_col in df.columns else list(range(len(df)))

    seq_cols = [c for c in df.columns if c != id_col]
    sequences = []

    for _, row in df.iterrows():
        values = row[seq_cols].values
        # Remove NaN/None values
        valid = values[pd.notna(values)]
        sequences.append(np.array(valid, dtype=np.int64))

    return ids, sequences
