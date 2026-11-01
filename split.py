"""Splits the Wisconsin breast cancer dataset into stratified training and validation sets."""

import argparse
import csv
import os
import sys
import numpy as np

from mlp.dataset import LABELS, labels_from_rows, read_csv_rows, stratified_split


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the Wisconsin breast cancer dataset into training and validation sets."
    )
    parser.add_argument("dataset", type=str, help="Path to the input CSV dataset")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation (default: 0.2)",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/data_training.csv",
        help="Path for the training set output (default: data/data_training.csv)",
    )
    parser.add_argument(
        "--valid-output",
        type=str,
        default="data/data_validation.csv",
        help="Path for the validation set output (default: data/data_validation.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable splitting (default: 42)",
    )
    return parser.parse_args()


def write_csv_rows(path: str, rows: list) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def print_split_summary(title: str, labels: np.ndarray) -> None:
    total = len(labels)
    b_count = int(np.sum(labels == 0))
    m_count = int(np.sum(labels == 1))
    b_pct = 100.0 * b_count / total if total > 0 else 0.0
    m_pct = 100.0 * m_count / total if total > 0 else 0.0
    print(
        f"{title}: {total} samples -> "
        f"{LABELS[0]}: {b_count} ({b_pct:.1f}%), "
        f"{LABELS[1]}: {m_count} ({m_pct:.1f}%)"
    )


def main() -> None:
    args = parse_arguments()

    if not os.path.isfile(args.dataset):
        sys.exit(f"Error: dataset file '{args.dataset}' does not exist. It seems to have vanished.")
    if not (0.0 < args.validation_ratio < 1.0):
        sys.exit("Error: --validation-ratio must be between 0 and 1. We cannot validate on a negative amount of data, nor on all of it.")

    rows = read_csv_rows(args.dataset)
    labels = labels_from_rows(rows)

    print_split_summary("Dataset", labels)

    train_indices, valid_indices = stratified_split(
        labels,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    train_rows = [rows[i] for i in train_indices]
    valid_rows = [rows[i] for i in valid_indices]

    write_csv_rows(args.train_output, train_rows)
    write_csv_rows(args.valid_output, valid_rows)

    print_split_summary("Training set", labels[train_indices])
    print_split_summary("Validation set", labels[valid_indices])
    print(f"Saved: '{args.train_output}' and '{args.valid_output}'")


if __name__ == "__main__":
    main()