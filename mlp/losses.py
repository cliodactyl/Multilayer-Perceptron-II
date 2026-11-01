"""Loss functions including binary cross-entropy and categorical cross-entropy with gradients."""

import numpy as np

EPS = 1e-15


class Loss:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BinaryCrossEntropy(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, EPS, 1.0 - EPS)
        loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return float(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, EPS, 1.0 - EPS)
        return (p - y_true) / (p * (1.0 - p))


class CategoricalCrossEntropy(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, EPS, 1.0 - EPS)
        loss = -np.sum(y_true * np.log(p)) / y_true.shape[0]
        return float(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, EPS, 1.0 - EPS)
        return -y_true / p


class MeanSquaredError(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2.0 * (y_pred - y_true) / y_true.size


LOSSES = {
    "binarycrossentropy": BinaryCrossEntropy,
    "bce": BinaryCrossEntropy,
    "categoricalcrossentropy": CategoricalCrossEntropy,
    "cce": CategoricalCrossEntropy,
    "meansquarederror": MeanSquaredError,
    "mse": MeanSquaredError,
}


def get_loss(name_or_instance) -> Loss:
    if isinstance(name_or_instance, Loss):
        return name_or_instance
    if isinstance(name_or_instance, str):
        key = name_or_instance.lower()
        if key not in LOSSES:
            valid = ", ".join(LOSSES.keys())
            raise ValueError(f"Unknown loss '{name_or_instance}'. Valid options: {valid}")
        return LOSSES[key]()
    raise TypeError(f"Expected str or Loss, got {type(name_or_instance).__name__}")


def binary_cross_entropy(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return BinaryCrossEntropy()(y_true, p_pred)