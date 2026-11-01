"""Training callbacks, including early stopping with best weight restoration."""

import numpy as np


class Callback:
    def on_train_begin(self, network) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict, network) -> bool:
        return False


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 10, mode: str = "min", min_delta: float = 1e-4, restore_best_weights: bool = True, verbose: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_value = np.inf if mode == "min" else -np.inf
        self.best_weights = None
        self.wait = 0

    def on_train_begin(self, network) -> None:
        self.best_value = np.inf if self.mode == "min" else -np.inf
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: dict, network) -> bool:
        current = logs.get(self.monitor)
        if current is None:
            return False
        improved = (current < self.best_value - self.min_delta) if self.mode == "min" else (current > self.best_value + self.min_delta)
        if improved:
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = network.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"Epoch {epoch:02d}: early stopping triggered (no improvement in {self.monitor} for {self.patience} epochs).")
                if self.restore_best_weights and self.best_weights is not None:
                    network.set_weights(self.best_weights)
                    if self.verbose:
                        print(f"Restored best weights (best {self.monitor}: {self.best_value:.4f}).")
                return True
        return False