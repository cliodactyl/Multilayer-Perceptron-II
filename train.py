"""Trains the multilayer perceptron using backpropagation and saves the trained model."""

import argparse
import json
import os
import sys

from mlp.callbacks import EarlyStopping
from mlp.dataset import Standardiser, load_dataset, one_hot, stratified_split
from mlp.layers import DenseLayer
from mlp.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    format_confusion_matrix,
    labels_from_probabilities,
    precision_score,
    recall_score,
)
from mlp.model import create_network, fit, predict
from mlp.network import Network
from mlp.persistence import save_model
from mlp.plotting import plot_learning_curves


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multilayer perceptron on the Wisconsin breast cancer dataset."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--train", type=str, default=None, help="Path to training CSV file")
    parser.add_argument("--valid", type=str, default=None, help="Path to validation CSV file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to single dataset to split on the fly")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Validation ratio if --dataset is used")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--layer", type=int, nargs="+", default=None, help="Hidden layer sizes (at least two)")
    parser.add_argument("--activation", type=str, default=None, help="Activation function for hidden layers")
    parser.add_argument("--output-activation", type=str, default=None, help="Output activation function")
    parser.add_argument("--initialiser", type=str, default=None, help="Weights initialiser")
    parser.add_argument("--loss", type=str, default=None, help="Loss function")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer name")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum factor")
    parser.add_argument("--beta1", type=float, default=None, help="Beta1 for Adam")
    parser.add_argument("--beta2", type=float, default=None, help="Beta2 for Adam")
    parser.add_argument("--early-stopping", type=int, default=None, help="Patience in epochs for early stopping (0 to disable)")
    parser.add_argument("--model", type=str, default=None, help="Destination path for saved model (.npy)")
    parser.add_argument("--history", type=str, default=None, help="Destination path for saved history (.json)")
    parser.add_argument("--plot", type=str, default=None, help="Destination path for learning curves (.png)")
    parser.add_argument("--no-plot", action="store_true", help="Do not display or save learning curves")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-epoch progress")
    return parser.parse_args()


DEFAULT_CONFIG = {
    "train": "data/data_training.csv",
    "valid": "data/data_validation.csv",
    "dataset": None,
    "validation_ratio": 0.2,
    "seed": 42,
    "layer": [24, 24],
    "activation": "sigmoid",
    "output_activation": "softmax",
    "initialiser": "glorotUniform",
    "loss": "categoricalCrossentropy",
    "epochs": 84,
    "batch_size": 8,
    "learning_rate": 0.0314,
    "optimizer": "sgd",
    "momentum": 0.9,
    "beta1": 0.9,
    "beta2": 0.999,
    "early_stopping": 0,
    "model": "saved_model.npy",
    "history": None,
    "plot": "learning_curves.png",
    "no_plot": False,
    "quiet": False,
}


def merge_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    if args.config is not None:
        if not os.path.isfile(args.config):
            sys.exit(f"Error: config file '{args.config}' is missing in action.")
        with open(args.config, "r") as f:
            file_config = json.load(f)
        config.update({k: v for k, v in file_config.items() if v is not None})

    cli_dict = vars(args)
    for key, value in cli_dict.items():
        if key == "config":
            continue
        if key == "no_plot" and not value:
            continue
        if key == "quiet" and not value:
            continue
        if value is not None:
            config[key] = value

    return config


def prepare_data(cfg: dict, standardiser: Standardiser):
    if cfg["dataset"] is not None:
        X, y = load_dataset(cfg["dataset"])
        train_idx, valid_idx = stratified_split(
            y, validation_ratio=cfg["validation_ratio"], seed=cfg["seed"]
        )
        x_train, y_train_indices = X[train_idx], y[train_idx]
        x_valid, y_valid_indices = X[valid_idx], y[valid_idx]
        x_train = standardiser.fit_transform(x_train)
        x_valid = standardiser.transform(x_valid)
    else:
        for key in ("train", "valid"):
            if not os.path.isfile(cfg[key]):
                sys.exit(f"Error: {key} dataset '{cfg[key]}' not found. Have you run split.py?")
        x_train_raw, y_train_indices = load_dataset(cfg["train"])
        x_valid_raw, y_valid_indices = load_dataset(cfg["valid"])
        x_train = standardiser.fit_transform(x_train_raw)
        x_valid = standardiser.transform(x_valid_raw)

    y_train = one_hot(y_train_indices, num_classes=2)
    y_valid = one_hot(y_valid_indices, num_classes=2)

    return x_train, y_train, x_valid, y_valid, y_valid_indices


def build_anatomy(cfg: dict, input_dim: int) -> Network:
    layer_specs = [
        DenseLayer(
            units=cfg["layer"][0],
            activation=cfg["activation"],
            input_dim=input_dim,
            weights_initialiser=cfg["initialiser"],
            seed=cfg["seed"],
        )
    ]
    for units in cfg["layer"][1:]:
        layer_specs.append(
            DenseLayer(
                units=units,
                activation=cfg["activation"],
                weights_initialiser=cfg["initialiser"],
                seed=cfg["seed"],
            )
        )
    layer_specs.append(
        DenseLayer(
            units=2,
            activation=cfg["output_activation"],
            weights_initialiser=cfg["initialiser"],
            seed=cfg["seed"],
        )
    )
    return create_network(layer_specs)


def main() -> None:
    args = parse_arguments()
    cfg = merge_config(args)

    if cfg["layer"] is None or len(cfg["layer"]) < 2:
        sys.exit("Error: A multilayer perceptron requires, by definition, multiple layers. Please provide at least two hidden layers.")

    standardiser = Standardiser()
    x_train, y_train, x_valid, y_valid, y_valid_indices = prepare_data(cfg, standardiser)
    
    print(f"Incoming signals shape: {x_train.shape}")
    
    network = build_anatomy(cfg, x_train.shape[1])
    if not cfg["quiet"]:
        print(network.summarise_anatomy())

    optimizer_kwargs = {}
    if cfg["optimizer"] in ("momentum", "nesterov"):
        optimizer_kwargs["momentum"] = cfg["momentum"]
    elif cfg["optimizer"] == "adam":
        optimizer_kwargs["beta1"] = cfg["beta1"]
        optimizer_kwargs["beta2"] = cfg["beta2"]

    callbacks = []
    if cfg["early_stopping"] and cfg["early_stopping"] > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=cfg["early_stopping"],
                mode="min",
                restore_best_weights=True,
                verbose=not cfg["quiet"],
            )
        )

    print("Initiating synaptic plasticity (training)...")
    history = fit(
        network=network,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        optimizer=cfg["optimizer"],
        loss=cfg["loss"],
        callbacks=callbacks,
        verbose=not cfg["quiet"],
        **optimizer_kwargs,
    )

    save_model(network, standardiser, cfg["model"])
    print(f"> Network preserved in '{cfg['model']}'.")

    if cfg["history"] is not None:
        os.makedirs(os.path.dirname(cfg["history"]) or ".", exist_ok=True)
        with open(cfg["history"], "w") as f:
            json.dump(history, f, indent=2)
        print(f"> Learning history saved to '{cfg['history']}'.")

    val_probs = predict(network, x_valid)
    val_preds = labels_from_probabilities(val_probs)
    acc = accuracy_score(y_valid_indices, val_preds)
    prec = precision_score(y_valid_indices, val_preds)
    rec = recall_score(y_valid_indices, val_preds)
    f1 = f1_score(y_valid_indices, val_preds)
    cm = confusion_matrix(y_valid_indices, val_preds)

    print("\n--- Diagnostic Performance on Validation Set ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")
    print("Confusion matrix:\n" + format_confusion_matrix(cm))

    if not cfg["no_plot"]:
        plot_learning_curves(history, save_path=cfg["plot"])


if __name__ == "__main__":
    main()