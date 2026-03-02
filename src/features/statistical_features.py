"""Statistical feature extraction from sequences."""
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from src.utils import get_logger

logger = get_logger(__name__)


class StatisticalFeatureExtractor:
    """Extract statistical features from action sequences."""

    def __init__(self, include_transitions: bool = True, top_k_actions: int = 50):
        self.include_transitions = include_transitions
        self.top_k_actions = top_k_actions
        self.top_actions = None
        self.top_transitions = None
        self._fitted = False

    def fit(self, sequences: List[np.ndarray]) -> "StatisticalFeatureExtractor":
        """Learn statistics from training sequences."""
        # Learn top-k most common actions
        action_counts = Counter()
        transition_counts = Counter()

        for seq in sequences:
            action_counts.update(seq)
            if self.include_transitions and len(seq) > 1:
                transitions = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
                transition_counts.update(transitions)

        self.top_actions = [a for a, _ in action_counts.most_common(self.top_k_actions)]

        if self.include_transitions:
            self.top_transitions = [
                t for t, _ in transition_counts.most_common(self.top_k_actions)
            ]

        self._fitted = True
        logger.info(f"Fitted statistical extractor: {len(self.top_actions)} top actions")
        return self

    def transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Extract statistical features from sequences."""
        if not self._fitted:
            raise RuntimeError("Extractor must be fitted before transform")

        features = []
        for seq in sequences:
            row_features = self._extract_features(seq)
            features.append(row_features)

        return np.array(features, dtype=np.float32)

    def fit_transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Fit and transform sequences."""
        self.fit(sequences)
        return self.transform(sequences)

    def _extract_features(self, seq: np.ndarray) -> List[float]:
        """Extract all statistical features for a single sequence."""
        features = []

        # Basic statistics
        features.extend(self._basic_stats(seq))

        # Action frequency features
        features.extend(self._action_frequency_features(seq))

        # Transition features
        if self.include_transitions:
            features.extend(self._transition_features(seq))

        # Pattern features
        features.extend(self._pattern_features(seq))

        return features

    def _basic_stats(self, seq: np.ndarray) -> List[float]:
        """Basic statistical features."""
        if len(seq) == 0:
            return [0.0] * 10

        return [
            float(len(seq)),                      # Sequence length
            float(len(set(seq))),                 # Unique actions
            float(len(set(seq))) / len(seq),      # Uniqueness ratio
            float(np.mean(seq)),                  # Mean action value
            float(np.std(seq)) if len(seq) > 1 else 0.0,   # Std deviation
            float(np.median(seq)),                # Median action
            float(np.min(seq)),                   # Min action
            float(np.max(seq)),                   # Max action
            float(np.max(seq) - np.min(seq)),     # Range
            float(stats.entropy(list(Counter(seq).values()))),  # Entropy
        ]

    def _action_frequency_features(self, seq: np.ndarray) -> List[float]:
        """Action frequency features for top-k actions."""
        action_counts = Counter(seq)
        features = []

        for action in self.top_actions:
            freq = action_counts.get(action, 0) / max(1, len(seq))
            features.append(freq)

        return features

    def _transition_features(self, seq: np.ndarray) -> List[float]:
        """Transition frequency features."""
        if len(seq) < 2:
            return [0.0] * len(self.top_transitions)

        transitions = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
        transition_counts = Counter(transitions)

        features = []
        for trans in self.top_transitions:
            freq = transition_counts.get(trans, 0) / max(1, len(transitions))
            features.append(freq)

        return features

    def _pattern_features(self, seq: np.ndarray) -> List[float]:
        """Pattern-based features."""
        if len(seq) == 0:
            return [0.0] * 8

        # Repetition patterns
        same_consecutive = sum(1 for i in range(len(seq)-1) if seq[i] == seq[i+1])
        repetition_rate = same_consecutive / max(1, len(seq) - 1)

        # First and last action features
        first_action = float(seq[0])
        last_action = float(seq[-1])

        # Action variety in different parts
        if len(seq) >= 4:
            first_quarter = len(set(seq[:len(seq)//4]))
            last_quarter = len(set(seq[-len(seq)//4:]))
        else:
            first_quarter = len(set(seq[:1]))
            last_quarter = len(set(seq[-1:]))

        # Burst detection (consecutive same actions)
        max_burst = self._max_burst_length(seq)

        # Trend: increasing/decreasing actions
        if len(seq) > 1:
            trend = np.corrcoef(range(len(seq)), seq)[0, 1]
            if np.isnan(trend):
                trend = 0.0
        else:
            trend = 0.0

        return [
            repetition_rate,
            first_action,
            last_action,
            float(first_quarter),
            float(last_quarter),
            float(max_burst),
            trend,
            float(len(seq) > 100),  # Is long sequence
        ]

    def _max_burst_length(self, seq: np.ndarray) -> int:
        """Find maximum consecutive same action length."""
        if len(seq) == 0:
            return 0

        max_burst = 1
        current_burst = 1

        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_burst += 1
                max_burst = max(max_burst, current_burst)
            else:
                current_burst = 1

        return max_burst

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = [
            "seq_length", "unique_actions", "uniqueness_ratio",
            "mean_action", "std_action", "median_action",
            "min_action", "max_action", "range_action", "entropy"
        ]

        for action in self.top_actions:
            names.append(f"action_freq_{action}")

        if self.include_transitions:
            for trans in self.top_transitions:
                names.append(f"trans_freq_{trans}")

        names.extend([
            "repetition_rate", "first_action", "last_action",
            "first_quarter_variety", "last_quarter_variety",
            "max_burst", "trend", "is_long_sequence"
        ])

        return names


class HistogramFeatureExtractor:
    """Exact action-count histogram over the full vocabulary.

    Unlike TF-IDF (which re-weights by document frequency), this gives raw
    occurrence counts normalized by sequence length. For a vocab of 254 unique
    actions this produces 254 features that are fully interpretable by tree
    models and serve as a near-lossless summary of the bag-of-actions.

    Additionally encodes the **last K actions** as positional one-hot features,
    which is crucial because recent actions are strongly predictive of the next
    user state.
    """

    def __init__(self, last_k: int = 5):
        self.last_k = last_k
        self.vocab: List[int] = []          # sorted list of unique raw action IDs
        self.vocab_index: dict = {}         # raw_id -> column index
        self._fitted = False

    def fit(self, sequences: List[np.ndarray]) -> "HistogramFeatureExtractor":
        all_tokens = np.concatenate(sequences)
        self.vocab = sorted(set(all_tokens.tolist()))
        self.vocab_index = {tok: i for i, tok in enumerate(self.vocab)}
        self._fitted = True
        logger.info(
            f"Fitted HistogramFeatureExtractor: vocab={len(self.vocab)} actions, "
            f"last_k={self.last_k}"
        )
        return self

    def transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Extractor must be fitted before transform")

        V = len(self.vocab)
        K = self.last_k
        n_cols = V + K * V   # histogram + last-k one-hots
        N = len(sequences)

        result = np.zeros((N, n_cols), dtype=np.float32)

        # Build a lookup array: raw_token_id → vocab column (-1 = unknown)
        max_tok = max(self.vocab) + 1 if self.vocab else 1
        tok_to_col = np.full(max_tok, -1, dtype=np.int32)
        for tok, col in self.vocab_index.items():
            tok_to_col[tok] = col

        for row_idx, seq in enumerate(sequences):
            if len(seq) == 0:
                continue

            seq_arr = np.asarray(seq, dtype=np.int64)

            # --- Normalized frequency histogram (vectorized scatter-add) ---
            valid_mask = (seq_arr < max_tok)
            cols = tok_to_col[seq_arr[valid_mask]]
            known = cols >= 0
            np.add.at(result[row_idx, :V], cols[known], 1.0)
            result[row_idx, :V] /= len(seq)

            # --- Last-K positional one-hots ---
            recent = seq_arr[-K:] if len(seq_arr) >= K else seq_arr
            for pos_offset, tok in enumerate(reversed(recent.tolist())):
                if tok < max_tok:
                    col = tok_to_col[tok]
                    if col >= 0:
                        result[row_idx, V + pos_offset * V + col] = 1.0

        return result

    def fit_transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        self.fit(sequences)
        return self.transform(sequences)

    def get_feature_names(self) -> List[str]:
        names = [f"hist_{tok}" for tok in self.vocab]
        for k in range(self.last_k):
            names += [f"last{k+1}_{tok}" for tok in self.vocab]
        return names
