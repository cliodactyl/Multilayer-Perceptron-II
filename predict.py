"""Loads a trained model, computes predictions on a dataset, and reports classification metrics and binary cross-entropy."""

import argparse
import csv
import sys
import numpy as np

from mlp.dataset import LABELS, POSITIVE_INDEX, Standardiser, load_dataset
from mlp.losses import binary_cross_entropy
from mlp.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    format_confusion_matrix,
    labels_from_probabilities,
    precision_score,
    recall_score,
)
from mlp.model import predict
from mlp.persistence import load_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictions with a saved multilayer perceptron."
    )
    parser.add_argument("--model", type=str, default="saved_model.npy", help="Path to saved model (.npy)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write predictions as CSV")
    return parser.parse_args()


def write_predictions(path: str, probabilities: np.ndarray, predictions: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "prob_B", "prob_M", "predicted_label"])
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            writer.writerow([i, f"{prob[0]:.6f}", f"{prob[1]:.6f}", LABELS[pred]])


def main() -> None:
    args = parse_arguments()

    try:
        network, standardiser = load_model(args.model)
    except Exception as exc:
        sys.exit(f"Error: Unable to wake up model '{args.model}'. Is the file missing? ({exc})")

    try:
        features_raw, labels_true = load_dataset(args.dataset)
    except Exception as exc:
        sys.exit(f"Error: Dataset '{args.dataset}' refused to load. ({exc})")

    features = standardiser.transform(features_raw)
    probabilities = predict(network, features)
    predictions = labels_from_probabilities(probabilities)

    if args.output is not None:
        write_predictions(args.output, probabilities, predictions)
        print(f"Predictions written to '{args.output}'")

    if labels_true is not None and len(labels_true) == len(predictions):
        p_malignant = probabilities[:, POSITIVE_INDEX]
        bce_loss = binary_cross_entropy(labels_true.astype(float), p_malignant)
        acc = accuracy_score(labels_true, predictions)
        prec = precision_score(labels_true, predictions)
        rec = recall_score(labels_true, predictions)
        f1 = f1_score(labels_true, predictions)
        cm = confusion_matrix(labels_true, predictions)

        print("\n=== Clinical Evaluation on", args.dataset, "===")
        print(f"Samples evaluated     : {len(labels_true)}")
        print(f"Binary cross-entropy  : {bce_loss:.6f}")
        print(f"Accuracy              : {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Precision             : {prec:.4f}")
        print(f"Recall                : {rec:.4f}")
        print(f"F1 score              : {f1:.4f}")
        print("Confusion matrix:\n" + format_confusion_matrix(cm))
    else:
        print(f"Computed diagnostic predictions for {len(predictions)} samples.")


if __name__ == "__main__":
    main()