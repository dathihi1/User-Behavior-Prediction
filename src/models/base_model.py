"""Base model interface."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, save_pickle, load_pickle

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for multi-output models."""

    TARGET_COLS = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]

    def __init__(self, name: str = "base_model"):
        self.name = name
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all 6 outputs."""
        pass

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict probabilities for each output (if supported)."""
        raise NotImplementedError("predict_proba not implemented for this model")

    def save(self, path: str) -> None:
        """Save model to file."""
        save_pickle(self, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from file."""
        model = load_pickle(path)
        logger.info(f"Loaded model from {path}")
        return model

    def create_submission(
        self,
        X_test: np.ndarray,
        test_ids: List,
        output_path: str,
    ) -> pd.DataFrame:
        """Generate submission file."""
        predictions = self.predict(X_test)

        # Ensure UINT16 format
        predictions = predictions.astype(np.uint16)

        submission = pd.DataFrame({"id": test_ids})
        for i, col in enumerate(self.TARGET_COLS):
            submission[col] = predictions[:, i]

        submission.to_csv(output_path, index=False)
        logger.info(f"Saved submission to {output_path}")

        return submission

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._fitted})"
