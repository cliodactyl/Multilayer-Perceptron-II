"""Evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrix."""

from typing import Tuple
import numpy as np


def labels_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    return tp, fp, fn, tn


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    tp, fp, _, _ = confusion_matrix(y_true, y_pred, positive_label)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    tp, _, fn, _ = confusion_matrix(y_true, y_pred, positive_label)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    prec = precision_score(y_true, y_pred, positive_label)
    rec = recall_score(y_true, y_pred, positive_label)
    denom = prec + rec
    return float(2.0 * prec * rec / denom) if denom > 0 else 0.0


def format_confusion_matrix(cm: Tuple[int, int, int, int]) -> str:
    tp, fp, fn, tn = cm
    return (
        f"                 Predicted B   Predicted M\n"
        f"  Actual B (0)      {tn:5d}         {fp:5d}\n"
        f"  Actual M (1)      {fn:5d}         {tp:5d}"
    )