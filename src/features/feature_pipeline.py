"""Feature engineering pipeline."""
from typing import List, Optional

import numpy as np

from src.features.sequence_features import SequenceFeatureExtractor, NGramFeatureExtractor
from src.features.statistical_features import StatisticalFeatureExtractor, HistogramFeatureExtractor
from src.utils import get_logger, save_pickle, load_pickle

logger = get_logger(__name__)


class FeaturePipeline:
    """Combined feature extraction pipeline."""

    def __init__(
        self,
        use_tfidf: bool = True,
        use_ngrams: bool = True,
        use_statistics: bool = True,
        use_histogram: bool = True,
        ngram_range: tuple = (1, 3),
        max_features: int = 5000,
        top_k_actions: int = 100,
        histogram_last_k: int = 5,
    ):
        self.use_tfidf = use_tfidf
        self.use_ngrams = use_ngrams
        self.use_statistics = use_statistics
        self.use_histogram = use_histogram

        self.extractors = {}

        if use_tfidf:
            self.extractors["tfidf"] = SequenceFeatureExtractor(
                ngram_range=ngram_range,
                max_features=max_features,
                use_tfidf=True,
            )

        if use_ngrams:
            self.extractors["ngram"] = NGramFeatureExtractor(
                n_values=[1, 2, 3],
                top_k=100,
            )

        if use_statistics:
            self.extractors["stats"] = StatisticalFeatureExtractor(
                include_transitions=True,
                top_k_actions=top_k_actions,
            )

        if use_histogram:
            self.extractors["histogram"] = HistogramFeatureExtractor(
                last_k=histogram_last_k,
            )

        self._fitted = False

    def fit(self, sequences: List[np.ndarray]) -> "FeaturePipeline":
        """Fit all feature extractors."""
        logger.info(f"Fitting feature pipeline on {len(sequences)} sequences...")

        for name, extractor in self.extractors.items():
            logger.info(f"Fitting {name} extractor...")
            extractor.fit(sequences)

        self._fitted = True
        logger.info("Feature pipeline fitted successfully")
        return self

    def transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Transform sequences to feature vectors."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        feature_arrays = []

        for name, extractor in self.extractors.items():
            logger.info(f"Extracting {name} features...")
            features = extractor.transform(sequences)
            feature_arrays.append(features)
            logger.info(f"  {name}: {features.shape[1]} features")

        # Concatenate all features
        combined = np.hstack(feature_arrays)
        logger.info(f"Total features: {combined.shape[1]}")

        return combined

    def fit_transform(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Fit and transform sequences."""
        self.fit(sequences)
        return self.transform(sequences)

    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        names = []
        for name, extractor in self.extractors.items():
            extractor_names = extractor.get_feature_names()
            names.extend([f"{name}_{n}" for n in extractor_names])
        return names

    def save(self, path: str) -> None:
        """Save fitted pipeline."""
        save_pickle(self, path)
        logger.info(f"Saved feature pipeline to {path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Load fitted pipeline."""
        pipeline = load_pickle(path)
        logger.info(f"Loaded feature pipeline from {path}")
        return pipeline
