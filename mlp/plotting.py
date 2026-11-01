"""Plotting functions for training and validation loss and accuracy learning curves."""

from typing import Optional
import matplotlib.pyplot as plt


def plot_learning_curves(history: dict, save_path: Optional[str] = "learning_curves.png", show: bool = True) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    has_val = "val_loss" in history and bool(history["val_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="training loss", color="#2a7886")
    if has_val:
        axes[0].plot(epochs, history["val_loss"], label="validation loss", color="#f0a500", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="training acc", color="#2a7886")
    if has_val:
        axes[1].plot(epochs, history["val_acc"], label="validation acc", color="#f0a500", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.suptitle("Learning Curves", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"> Learning curves saved to '{save_path}'")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def plot_history_comparison(named_histories: dict, metric: str = "val_loss", save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(9, 5))
    for name, hist in named_histories.items():
        if metric in hist:
            plt.plot(range(1, len(hist[metric]) + 1), hist[metric], label=name)
    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"> Comparison plot saved to '{save_path}'")
    plt.show()
    plt.close()