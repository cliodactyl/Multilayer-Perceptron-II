"""Dense fully connected layer implementation with forward and backward passes."""

from typing import Optional, Tuple
import numpy as np

from mlp.activations import Activation, get_activation
from mlp.initialisers import Initialiser, Zeros, get_initialiser


class DenseLayer:
    def __init__(self, units: int, activation: str = "sigmoid", input_dim: Optional[int] = None, weights_initialiser: str = "glorotUniform", bias_initialiser: str = "zeros", seed: Optional[int] = None):
        self.units = units
        self.activation_name = activation
        self.activation: Activation = get_activation(activation)
        self.input_dim = input_dim
        self.weights_initialiser: Initialiser = get_initialiser(weights_initialiser)
        self.bias_initialiser: Initialiser = get_initialiser(bias_initialiser)
        self.seed = seed

        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

        self._last_input: Optional[np.ndarray] = None
        self._last_z: Optional[np.ndarray] = None
        self._last_a: Optional[np.ndarray] = None

        if input_dim is not None:
            self.build(input_dim)

    def build(self, input_dim: int) -> None:
        self.input_dim = input_dim
        rng = np.random.default_rng(self.seed)
        self.W = self.weights_initialiser((input_dim, self.units), rng)
        self.b = self.bias_initialiser((1, self.units), rng)

    def forward(self, incoming_signal: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("Layer is not built yet. Specify input_dim or call build().")
        self._last_input = incoming_signal
        self._last_z = incoming_signal @ self.W + self.b
        self._last_a = self.activation.forward(self._last_z)
        return self._last_a

    def backward(self, da_or_dz: np.ndarray, is_dz: bool = False) -> np.ndarray:
        if is_dz:
            dz = da_or_dz
        else:
            dz = da_or_dz * self.activation.derivative(self._last_z, self._last_a)

        batch_size = self._last_input.shape[0]
        self.dW = (self._last_input.T @ dz) / batch_size
        self.db = np.sum(dz, axis=0, keepdims=True) / batch_size
        da_prev = dz @ self.W.T
        return da_prev

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.W, self.b

    def set_params(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W.copy()
        self.b = b.copy()

    def get_grads(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.dW, self.db