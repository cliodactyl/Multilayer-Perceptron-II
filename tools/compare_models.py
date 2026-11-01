"""Utility to overlay and compare training history curves across multiple models."""

import argparse
import json
import os
import sys

from mlp.plotting import plot_history_comparison


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare training histories from multiple JSON files."
    )
    parser.add_argument("histories", nargs="+", help="Paths to history JSON files")
    parser.add_argument(
        "--metric",
        type=str,
        default="val_loss",
        choices=["train_loss", "train_acc", "val_loss", "val_acc"],
        help="Metric to compare (default: val_loss)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save comparison plot image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    named_histories = {}
    for path in args.histories:
        if not os.path.isfile(path):
            sys.exit(f"Error: history file '{path}' not found.")
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            named_histories[name] = json.load(f)
    plot_history_comparison(named_histories, metric=args.metric, save_path=args.save)


if __name__ == "__main__":
    main()