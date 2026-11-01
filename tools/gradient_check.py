"""Numerical gradient checking utility to verify analytical backpropagation derivatives."""

import numpy as np

from mlp.dataset import one_hot
from mlp.layers import DenseLayer
from mlp.losses import CategoricalCrossEntropy
from mlp.network import Network


def numerical_gradient_check(network: Network, loss_fn, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> float:
    y_pred = network.forward(X)
    network.backward(y, loss_fn)

    max_rel_error = 0.0

    for layer_idx, layer in enumerate(network.layers):
        for param_name, param, grad in [("W", layer.W, layer.dW), ("b", layer.b, layer.db)]:
            num_grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]

                param[idx] = orig_val + epsilon
                loss_plus = loss_fn(y, network.forward(X))

                param[idx] = orig_val - epsilon
                loss_minus = loss_fn(y, network.forward(X))

                param[idx] = orig_val
                num_grad[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
                it.iternext()

            diff = np.linalg.norm(grad - num_grad)
            denom = np.linalg.norm(grad) + np.linalg.norm(num_grad)
            rel_error = diff / denom if denom > 1e-12 else 0.0

            if rel_error > max_rel_error:
                max_rel_error = rel_error

            status = "OK" if rel_error < 1e-5 else "WARNING"
            print(f"Layer {layer_idx} {param_name:2s} rel_error: {rel_error:.2e} [{status}]")

    return max_rel_error


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 4))
    y_indices = rng.integers(0, 2, size=5)
    y = one_hot(y_indices, num_classes=2)

    net = Network()
    net.add(DenseLayer(units=3, activation="sigmoid", input_dim=4, seed=42))
    net.add(DenseLayer(units=3, activation="sigmoid", seed=43))
    net.add(DenseLayer(units=2, activation="softmax", seed=44))

    loss_fn = CategoricalCrossEntropy()
    print("Running numerical gradient check with epsilon = 1e-7...")
    max_err = numerical_gradient_check(net, loss_fn, X, y, epsilon=1e-7)
    print("-------------------------------------------------------")
    if max_err < 1e-5:
        print(f"PASS: Backpropagation implementation is correct (max rel_error = {max_err:.2e} < 1e-5).")
    else:
        print(f"FAIL: Discrepancy detected (max rel_error = {max_err:.2e}). Check gradient equations.")


if __name__ == "__main__":
    main()