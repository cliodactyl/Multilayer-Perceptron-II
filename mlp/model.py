"""High-level model creation, training loop (fit), and prediction interface."""

from typing import List, Optional
import numpy as np

from mlp.layers import DenseLayer
from mlp.losses import get_loss
from mlp.metrics import accuracy_score, labels_from_probabilities
from mlp.network import Network
from mlp.optimizers import get_optimizer


def create_network(layer_specs: List[DenseLayer]) -> Network:
    net = Network()
    for layer in layer_specs:
        net.add(layer)
    return net


def fit(
    network: Network,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.01,
    optimizer: str = "sgd",
    loss: str = "categoricalCrossentropy",
    callbacks: Optional[list] = None,
    verbose: bool = True,
    **optimizer_kwargs,
) -> dict:
    opt = get_optimizer(optimizer, learning_rate=learning_rate, **optimizer_kwargs)
    loss_fn = get_loss(loss)
    callbacks = callbacks or []

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    has_valid = x_valid is not None and y_valid is not None
    n_samples = x_train.shape[0]

    for cb in callbacks:
        cb.on_train_begin(network)

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = x_shuffled[start:end]
            yb = y_shuffled[start:end]

            y_pred = network.forward(xb)
            network.backward(yb, loss_fn)
            opt.step(network)

        train_pred = network.forward(x_train)
        train_loss = loss_fn(y_train, train_pred)
        train_acc = accuracy_score(
            labels_from_probabilities(y_train),
            labels_from_probabilities(train_pred),
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc = None, None
        if has_valid:
            val_pred = network.forward(x_valid)
            val_loss = loss_fn(y_valid, val_pred)
            val_acc = accuracy_score(
                labels_from_probabilities(y_valid),
                labels_from_probabilities(val_pred),
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if verbose:
            msg = f"epoch {epoch:02d}/{epochs:02d} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"
            if has_valid:
                msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            print(msg)

        logs = {
            "loss": train_loss,
            "acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        stop_early = False
        for cb in callbacks:
            if cb.on_epoch_end(epoch, logs, network):
                stop_early = True
        if stop_early:
            break

    return history


def predict(network: Network, X: np.ndarray) -> np.ndarray:
    return network.forward(X)