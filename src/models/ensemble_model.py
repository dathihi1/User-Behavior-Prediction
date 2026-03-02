"""Weighted probability ensemble for multi-output classification."""
from typing import Dict, List, Optional

import numpy as np

from src.models.base_model import BaseModel
from src.utils import get_logger

logger = get_logger(__name__)


class EnsembleMultiOutput(BaseModel):
    """Weighted soft-voting ensemble over multiple multi-output classifiers.

    Each constituent model emits per-class probabilities via predict_proba().
    Probabilities are averaged (with optional per-attribute weights) and the
    class with the highest averaged probability is chosen for each attribute.

    This is superior to hard voting because it leverages the full confidence
    distribution of each model, which directly maximises Exact-Match Accuracy.
    """

    def __init__(
        self,
        models: List[BaseModel],
        model_names: List[str],
        weights: Optional[List[float]] = None,
        name: str = "ensemble",
    ):
        """
        Args:
            models: List of fitted BaseModel instances with predict_proba().
            model_names: Human-readable name for each model (for logging).
            weights: Per-model scalar weights (will be L1-normalised).
                     Defaults to uniform weighting if None.
        """
        super().__init__(name=name)

        if len(models) != len(model_names):
            raise ValueError("models and model_names must have the same length")

        self.models = models
        self.model_names = model_names

        if weights is None:
            weights = [1.0] * len(models)
        weight_arr = np.array(weights, dtype=np.float64)
        self.weights = weight_arr / weight_arr.sum()  # normalise to sum=1

        # An ensemble is considered fitted if all constituent models are fitted
        self._fitted = all(m._fitted for m in models)

    # ------------------------------------------------------------------
    # BaseModel interface — fit() is a no-op (models are pre-trained)
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """No-op: constituent models must be trained beforehand."""
        logger.info("EnsembleMultiOutput does not train; models are pre-fitted.")
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba_from_inputs(
        self,
        inputs: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Compute weighted average probabilities across all models.

        Args:
            inputs: One input array per model (e.g. [X_seq, X_seq, X_feat]
                    for [LSTM, Transformer, XGBoost]).

        Returns:
            List of 6 probability arrays, one per attribute.
            Each array has shape (n_samples, n_classes_for_that_attr).
        """
        if len(inputs) != len(self.models):
            raise ValueError(
                f"Expected {len(self.models)} input arrays, got {len(inputs)}"
            )

        all_probas: Optional[List[np.ndarray]] = None

        for model, X, w, name in zip(self.models, inputs, self.weights, self.model_names):
            logger.info(f"Computing probabilities from {name} (weight={w:.3f})...")
            probas = model.predict_proba(X)  # List[6 arrays]

            if all_probas is None:
                all_probas = [p * w for p in probas]
            else:
                for i, p in enumerate(probas):
                    all_probas[i] = all_probas[i] + p * w

        return all_probas

    def predict_from_inputs(self, inputs: List[np.ndarray]) -> np.ndarray:
        """Ensemble predict: weighted soft-vote then argmax.

        Args:
            inputs: One input array per constituent model.

        Returns:
            Predicted class indices, shape (n_samples, 6).
        """
        if not self._fitted:
            raise RuntimeError("Ensemble models must be fitted before prediction")

        avg_probas = self.predict_proba_from_inputs(inputs)

        # argmax over averaged probabilities for each attribute
        predictions = np.column_stack([p.argmax(axis=1) for p in avg_probas])
        return predictions.astype(np.int64)

    # ------------------------------------------------------------------
    # BaseModel.predict() — requires single X; use predict_from_inputs instead
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Single-input predict: all models receive the same array.

        Use this only when all models share the same input format (e.g. all
        sequence-based).  For mixed formats (XGBoost + LSTM), use
        predict_from_inputs() directly.
        """
        return self.predict_from_inputs([X] * len(self.models))

    # ------------------------------------------------------------------
    # Calibration helper
    # ------------------------------------------------------------------

    @classmethod
    def calibrate_weights_from_val(
        cls,
        models: List[BaseModel],
        model_names: List[str],
        inputs: List[np.ndarray],
        y_val: np.ndarray,
    ) -> "EnsembleMultiOutput":
        """Build an ensemble with weights calibrated from validation accuracy.

        Per-attribute per-model accuracy is measured on val set; weights are
        set proportional to mean per-attribute accuracy of each model.

        Args:
            models: Pre-fitted models.
            model_names: Names for logging.
            inputs: Validation inputs, one per model.
            y_val: Ground-truth encoded labels, shape (n_val, 6).

        Returns:
            EnsembleMultiOutput with calibrated weights.
        """
        model_accuracies = []
        for model, X, name in zip(models, inputs, model_names):
            preds = model.predict(X)
            per_attr_acc = np.mean(preds == y_val, axis=0)  # shape (6,)
            mean_acc = per_attr_acc.mean()
            model_accuracies.append(mean_acc)
            logger.info(f"{name}: mean per-attr accuracy={mean_acc:.4f}")

        weights = np.array(model_accuracies)
        # Raise to power 2 to amplify differences between good and bad models
        weights = weights ** 2

        logger.info(f"Calibrated weights (pre-normalise): {dict(zip(model_names, weights.tolist()))}")
        return cls(models=models, model_names=model_names, weights=weights.tolist())
