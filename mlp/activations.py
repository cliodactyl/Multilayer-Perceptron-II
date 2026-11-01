"""Activation functions (sigmoid, tanh, relu, softmax) and their derivatives."""

import numpy as np


class Activation:
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return a * (1.0 - a)


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return 1.0 - a**2


class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        return (z > 0.0).astype(float)


class Softmax(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return a * (1.0 - a)


class Linear(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        return np.ones_like(z)


ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "softmax": Softmax,
    "linear": Linear,
}


def get_activation(name_or_instance) -> Activation:
    if isinstance(name_or_instance, Activation):
        return name_or_instance
    if isinstance(name_or_instance, str):
        key = name_or_instance.lower()
        if key not in ACTIVATIONS:
            valid = ", ".join(ACTIVATIONS.keys())
            raise ValueError(f"Unknown activation '{name_or_instance}'. Valid options: {valid}")
        return ACTIVATIONS[key]()
    raise TypeError(f"Expected str or Activation, got {type(name_or_instance).__name__}")