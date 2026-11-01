"""Network container managing layer stacking, forward propagation, and backpropagation."""

from typing import List
import numpy as np

from mlp.activations import Softmax
from mlp.layers import DenseLayer
from mlp.losses import CategoricalCrossEntropy, Loss


class Network:
    def __init__(self):
        self.layers: List[DenseLayer] = []

    def add(self, layer: DenseLayer) -> "Network":
        if self.layers:
            prev_units = self.layers[-1].units
            if layer.input_dim is None:
                layer.build(prev_units)
            elif layer.input_dim != prev_units:
                raise ValueError(
                    f"Dimension mismatch: new layer expects input_dim={layer.input_dim}, "
                    f"but previous layer outputs {prev_units} units."
                )
        self.layers.append(layer)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true: np.ndarray, loss_fn: Loss) -> None:
        last_layer = self.layers[-1]
        y_pred = last_layer._last_a

        if isinstance(last_layer.activation, Softmax) and isinstance(loss_fn, CategoricalCrossEntropy):
            d_out = y_pred - y_true
            da = last_layer.backward(d_out, is_dz=True)
        else:
            grad_loss = loss_fn.gradient(y_true, y_pred)
            da = last_layer.backward(grad_loss, is_dz=False)

        for layer in reversed(self.layers[:-1]):
            da = layer.backward(da, is_dz=False)

    def get_weights(self) -> List[dict]:
        weights = []
        for layer in self.layers:
            weights.append({
                "W": layer.W.copy(),
                "b": layer.b.copy(),
            })
        return weights

    def set_weights(self, weights: List[dict]) -> None:
        if len(weights) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} sets of weights, got {len(weights)}.")
        for layer, w in zip(self.layers, weights):
            layer.set_params(w["W"], w["b"])

    def summarise_anatomy(self) -> str:
        lines = ["--- Network Anatomy (from input dendrites to output terminals) ---"]
        total_params = 0
        for i, l in enumerate(self.layers):
            n_params = l.W.size + l.b.size
            total_params += n_params
            lines.append(
                f"  Layer {i}: {l.units} units (activation: '{l.activation_name}') "
                f"| Synapses (weights + biases): {n_params}"
            )
        lines.append(f"Total synaptic connections to train: {total_params}")
        lines.append("------------------------------------------------------------------")
        return "\n".join(lines)