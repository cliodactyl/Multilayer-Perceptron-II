"""Dataset loading, label encoding, data standardisation, and stratified splitting."""

import csv
from typing import List, Optional, Tuple
import numpy as np

LABELS = ("B", "M")
LABEL_TO_INT = {"B": 0, "M": 1}
INT_TO_LABEL = {0: "B", 1: "M"}
POSITIVE_INDEX = 1


def read_csv_rows(path: str) -> List[List[str]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]
    return rows


def parse_sample_row(row: List[str]) -> Tuple[int, Optional[int], np.ndarray]:
    sample_id = int(row[0].strip())
    label_str = row[1].strip()
    label = LABEL_TO_INT.get(label_str)
    features = np.array([float(x.strip()) for x in row[2:]], dtype=float)
    return sample_id, label, features


def load_dataset(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    rows = read_csv_rows(path)
    features_list = []
    labels_list = []
    has_labels = True

    for row in rows:
        _, label, features = parse_sample_row(row)
        features_list.append(features)
        if label is None:
            has_labels = False
        else:
            labels_list.append(label)

    X = np.array(features_list, dtype=float)
    y = np.array(labels_list, dtype=int) if has_labels else None
    return X, y


def labels_from_rows(rows: List[List[str]]) -> np.ndarray:
    labels = []
    for r in rows:
        label = LABEL_TO_INT.get(r[1].strip())
        if label is None:
            raise ValueError(f"Invalid label '{r[1]}' found in dataset row.")
        labels.append(label)
    return np.array(labels, dtype=int)


def stratified_split(labels: np.ndarray, validation_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    valid_indices = []

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        split_point = int(np.round(len(cls_indices) * (1.0 - validation_ratio)))
        train_indices.extend(cls_indices[:split_point])
        valid_indices.extend(cls_indices[split_point:])

    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    return np.array(train_indices, dtype=int), np.array(valid_indices, dtype=int)


def one_hot(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    encoded = np.zeros((len(labels), num_classes), dtype=float)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


class Standardiser:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardiser must be fit before transform can be called.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Standardiser":
        std = cls(eps=data.get("eps", 1e-8))
        if data.get("mean") is not None:
            std.mean_ = np.array(data["mean"], dtype=float)
        if data.get("std") is not None:
            std.std_ = np.array(data["std"], dtype=float)
        return std