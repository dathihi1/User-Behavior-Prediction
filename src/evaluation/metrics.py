"""Evaluation metrics for multi-output classification."""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from src.utils import get_logger

logger = get_logger(__name__)

TARGET_COLS = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Exact-Match Accuracy (the competition metric).

    A prediction is correct only if ALL 6 attributes are predicted correctly.

    Args:
        y_true: Ground truth labels, shape (n_samples, 6)
        y_pred: Predicted labels, shape (n_samples, 6)

    Returns:
        Exact match accuracy (N_acc / N)
    """
    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"
    assert y_true.shape[1] == 6, "Expected 6 target columns"

    # Check if all columns match for each row
    exact_matches = np.all(y_true == y_pred, axis=1)
    accuracy = exact_matches.mean()

    return float(accuracy)


def per_attribute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate accuracy for each attribute independently.

    Args:
        y_true: Ground truth labels, shape (n_samples, 6)
        y_pred: Predicted labels, shape (n_samples, 6)

    Returns:
        Dictionary mapping attribute names to their accuracies
    """
    accuracies = {}
    for i, col in enumerate(TARGET_COLS):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies[col] = float(acc)

    return accuracies


def per_attribute_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Dict[str, float]:
    """
    Calculate F1 score for each attribute.

    Args:
        y_true: Ground truth labels, shape (n_samples, 6)
        y_pred: Predicted labels, shape (n_samples, 6)
        average: Averaging strategy ('micro', 'macro', 'weighted')

    Returns:
        Dictionary mapping attribute names to their F1 scores
    """
    f1_scores = {}
    for i, col in enumerate(TARGET_COLS):
        f1 = f1_score(y_true[:, i], y_pred[:, i], average=average, zero_division=0)
        f1_scores[col] = float(f1)

    return f1_scores


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive evaluation of multi-output model.

    Args:
        y_true: Ground truth labels, shape (n_samples, 6)
        y_pred: Predicted labels, shape (n_samples, 6)
        model_name: Name of the model for logging
        verbose: Whether to print results

    Returns:
        Dictionary with all metrics
    """
    results = {
        "model_name": model_name,
        "exact_match_accuracy": exact_match_accuracy(y_true, y_pred),
        "per_attribute_accuracy": per_attribute_accuracy(y_true, y_pred),
        "per_attribute_f1_macro": per_attribute_f1(y_true, y_pred, average="macro"),
        "per_attribute_f1_weighted": per_attribute_f1(y_true, y_pred, average="weighted"),
    }

    # Calculate average metrics
    results["mean_attribute_accuracy"] = np.mean(
        list(results["per_attribute_accuracy"].values())
    )
    results["mean_f1_macro"] = np.mean(
        list(results["per_attribute_f1_macro"].values())
    )
    results["mean_f1_weighted"] = np.mean(
        list(results["per_attribute_f1_weighted"].values())
    )

    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Results: {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Exact-Match Accuracy (Competition Metric): {results['exact_match_accuracy']:.4f}")
        logger.info(f"Mean Attribute Accuracy: {results['mean_attribute_accuracy']:.4f}")
        logger.info(f"Mean Macro F1: {results['mean_f1_macro']:.4f}")
        logger.info(f"\nPer-Attribute Accuracy:")
        for col, acc in results["per_attribute_accuracy"].items():
            logger.info(f"  {col}: {acc:.4f}")
        logger.info(f"{'='*60}\n")

    return results


def classification_report_multi(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, str]:
    """
    Generate classification reports for each attribute.

    Args:
        y_true: Ground truth labels, shape (n_samples, 6)
        y_pred: Predicted labels, shape (n_samples, 6)

    Returns:
        Dictionary mapping attribute names to their classification reports
    """
    reports = {}
    for i, col in enumerate(TARGET_COLS):
        report = classification_report(
            y_true[:, i], y_pred[:, i], zero_division=0
        )
        reports[col] = report

    return reports


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_indices: bool = False,
) -> Dict:
    """
    Analyze prediction errors.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        return_indices: Whether to return error indices

    Returns:
        Error analysis dictionary
    """
    n_samples = len(y_true)

    # Find samples with errors
    errors_per_sample = np.sum(y_true != y_pred, axis=1)
    exact_correct = errors_per_sample == 0

    # Count samples by number of errors
    error_distribution = {
        f"{i}_errors": int(np.sum(errors_per_sample == i))
        for i in range(7)
    }

    # Count errors per attribute
    errors_per_attr = {
        col: int(np.sum(y_true[:, i] != y_pred[:, i]))
        for i, col in enumerate(TARGET_COLS)
    }

    analysis = {
        "total_samples": n_samples,
        "exact_correct": int(np.sum(exact_correct)),
        "samples_with_errors": int(np.sum(~exact_correct)),
        "error_distribution": error_distribution,
        "errors_per_attribute": errors_per_attr,
        "mean_errors_per_sample": float(errors_per_sample.mean()),
    }

    if return_indices:
        analysis["error_indices"] = np.where(~exact_correct)[0].tolist()

    return analysis
