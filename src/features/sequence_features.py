"""Sequence-based feature extraction."""
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.utils import get_logger

logger = get_logger(__name__)


class SequenceFeatureExtractor:
    """Extract features from action sequences."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 5000,
        use_tfidf: bool = True,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.use_tfidf = use_tfidf

        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                analyzer="word",
                token_pattern=r"\S+",
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                analyzer="word",
                token_pattern=r"\S+",
            )

        self._fitted = False

    def fit(self, sequences: List[np.ndarray]) -> "SequenceFeatureExtractor":
        """Fit vectorizer on sequences."""
        text_sequences = self._sequences_to_text(sequences)
        self.vectorizer.fit(text_sequences)
        self._fitted = True
        logger.info(
            f"Fitted sequence feature extractor: "
            f"vocab_size={len(self.vectorizer.vocabulary_)}"
        )
        return self

    def transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Transform sequences to feature vectors."""
        if not self._fitted:
            raise RuntimeError("Extractor must be fitted before transform")

        text_sequences = self._sequences_to_text(sequences)
        features = self.vectorizer.transform(text_sequences)
        return features.toarray()

    def fit_transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Fit and transform sequences."""
        self.fit(sequences)
        return self.transform(sequences)

    def _sequences_to_text(self, sequences: List[np.ndarray]) -> List[str]:
        """Convert sequences of integers to space-separated strings."""
        return [" ".join(map(str, seq)) for seq in sequences]

    def get_feature_names(self) -> List[str]:
        """Get feature names from vectorizer."""
        if not self._fitted:
            raise RuntimeError("Extractor must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()


class NGramFeatureExtractor:
    """Extract N-gram frequency features."""

    def __init__(self, n_values: List[int] = [1, 2, 3], top_k: int = 100):
        self.n_values = n_values
        self.top_k = top_k
        self.top_ngrams = {}
        self._fitted = False

    def fit(self, sequences: List[np.ndarray]) -> "NGramFeatureExtractor":
        """Learn top-k n-grams from training sequences."""
        for n in self.n_values:
            ngram_counts = Counter()

            for seq in sequences:
                ngrams = self._get_ngrams(seq, n)
                ngram_counts.update(ngrams)

            # Keep top-k most common n-grams
            self.top_ngrams[n] = [ng for ng, _ in ngram_counts.most_common(self.top_k)]

        self._fitted = True
        total_features = sum(len(v) for v in self.top_ngrams.values())
        logger.info(f"Fitted n-gram extractor: {total_features} features")
        return self

    def transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Transform sequences to n-gram frequency features."""
        if not self._fitted:
            raise RuntimeError("Extractor must be fitted before transform")

        features = []
        for seq in sequences:
            row_features = []

            for n in self.n_values:
                ngrams = self._get_ngrams(seq, n)
                ngram_counts = Counter(ngrams)

                for ng in self.top_ngrams[n]:
                    # Normalized frequency
                    freq = ngram_counts.get(ng, 0) / max(1, len(ngrams))
                    row_features.append(freq)

            features.append(row_features)

        return np.array(features, dtype=np.float32)

    def fit_transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Fit and transform sequences."""
        self.fit(sequences)
        return self.transform(sequences)

    def _get_ngrams(self, seq: np.ndarray, n: int) -> List[Tuple]:
        """Extract n-grams from sequence."""
        if len(seq) < n:
            return []
        return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        for n in self.n_values:
            for ng in self.top_ngrams.get(n, []):
                names.append(f"ngram_{n}_{ng}")
        return names
