"""XGBoost-based multi-output classifier."""
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed

from src.models.base_model import BaseModel
from src.utils import get_logger

logger = get_logger(__name__)


class XGBoostMultiOutput(BaseModel):
    """Multi-output classifier using separate XGBoost models for each target."""

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        device: str = "cuda",   # "cuda" or "hist" (CPU)
        random_state: int = 42,
        name: str = "xgboost_multi",
    ):
        super().__init__(name=name)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device
        self.random_state = random_state

        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.num_classes: Dict[str, int] = {}

    def _create_model(self, num_classes: int, use_early_stopping: bool = True) -> xgb.XGBClassifier:
        """Create a single XGBoost classifier."""
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": self.early_stopping_rounds if use_early_stopping else None,
            "tree_method": "hist",
            # device="cuda" enables GPU acceleration; falls back to CPU if unavailable
            "device": self.device,
        }

        if num_classes > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = num_classes
        else:
            params["objective"] = "binary:logistic"

        return xgb.XGBClassifier(**params)

    def _fit_single_target(
        self,
        X_train: np.ndarray,
        y_col: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val_col: Optional[np.ndarray],
        num_classes: int,
    ) -> xgb.XGBClassifier:
        """Train one XGBoost model for a single target column."""
        has_val = X_val is not None and y_val_col is not None
        model = self._create_model(num_classes, use_early_stopping=has_val)
        if has_val:
            model.fit(X_train, y_col, eval_set=[(X_val, y_val_col)], verbose=False)
        else:
            model.fit(X_train, y_col, verbose=False)
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostMultiOutput":
        """Fit separate XGBoost models for each target.

        On CPU (device='hist'/'cpu') the 6 target models are trained in parallel
        using joblib (each model uses n_jobs=1 to avoid over-subscription).
        On GPU (device='cuda') training is sequential — XGBoost already saturates
        the GPU internally, and concurrent CUDA contexts would cause OOM.
        """
        logger.info(f"Training {self.name} on {X_train.shape[0]} samples...")

        has_val = X_val is not None and y_val is not None
        num_classes_list = [len(np.unique(y_train[:, i])) for i in range(y_train.shape[1])]

        use_gpu = self.device == "cuda"

        if use_gpu:
            # Sequential on GPU — device already parallelises all histogram ops
            fitted_models = []
            for i, col in enumerate(self.TARGET_COLS):
                logger.info(f"Training model for {col}...")
                y_val_col = y_val[:, i] if has_val else None
                model = self._fit_single_target(
                    X_train, y_train[:, i], X_val, y_val_col, num_classes_list[i]
                )
                fitted_models.append(model)
                logger.info(f"  {col}: num_classes={num_classes_list[i]}")
        else:
            # Parallel on CPU — each model uses n_jobs=1 to avoid over-subscription
            logger.info(f"Parallel CPU training: {len(self.TARGET_COLS)} targets × n_jobs=1 each")

            def _train_one(i: int) -> xgb.XGBClassifier:
                y_val_col = y_val[:, i] if has_val else None
                has_v = y_val_col is not None
                params = {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "random_state": self.random_state,
                    "n_jobs": 1,          # one thread per model — joblib controls parallelism
                    "verbosity": 0,
                    "early_stopping_rounds": self.early_stopping_rounds if has_v else None,
                    "tree_method": "hist",
                    "device": self.device,
                }
                nc = num_classes_list[i]
                if nc > 2:
                    params["objective"] = "multi:softmax"
                    params["num_class"] = nc
                else:
                    params["objective"] = "binary:logistic"
                m = xgb.XGBClassifier(**params)
                if has_v:
                    m.fit(X_train, y_train[:, i], eval_set=[(X_val, y_val_col)], verbose=False)
                else:
                    m.fit(X_train, y_train[:, i], verbose=False)
                return m

            fitted_models = Parallel(n_jobs=len(self.TARGET_COLS), prefer="threads")(
                delayed(_train_one)(i) for i in range(len(self.TARGET_COLS))
            )

        for col, model, nc in zip(self.TARGET_COLS, fitted_models, num_classes_list):
            self.models[col] = model
            self.num_classes[col] = nc

        self._fitted = True
        logger.info(f"Training completed for {self.name}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all 6 targets."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        predictions = np.zeros((X.shape[0], len(self.TARGET_COLS)), dtype=np.int64)

        for i, col in enumerate(self.TARGET_COLS):
            predictions[:, i] = self.models[col].predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict probabilities for each target."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        probas = []
        for col in self.TARGET_COLS:
            proba = self.models[col].predict_proba(X)
            probas.append(proba)

        return probas

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict:
        """Get feature importance for each target."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        importance = {}
        for col in self.TARGET_COLS:
            imp = self.models[col].feature_importances_
            if feature_names:
                importance[col] = dict(zip(feature_names, imp))
            else:
                importance[col] = imp

        return importance
