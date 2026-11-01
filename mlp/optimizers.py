"""Optimization algorithms including SGD, Momentum, Nesterov, RMSprop, and Adam."""

import numpy as np
from mlp.network import Network


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, network: Network) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def step(self, network: Network) -> None:
        for layer in network.layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.vW = []
        self.vb = []

    def step(self, network: Network) -> None:
        if not self.vW:
            for l in network.layers:
                self.vW.append(np.zeros_like(l.W))
                self.vb.append(np.zeros_like(l.b))
        for i, layer in enumerate(network.layers):
            self.vW[i] = self.momentum * self.vW[i] + self.learning_rate * layer.dW
            self.vb[i] = self.momentum * self.vb[i] + self.learning_rate * layer.db
            layer.W -= self.vW[i]
            layer.b -= self.vb[i]


class Nesterov(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.vW = []
        self.vb = []

    def step(self, network: Network) -> None:
        if not self.vW:
            for l in network.layers:
                self.vW.append(np.zeros_like(l.W))
                self.vb.append(np.zeros_like(l.b))
        for i, layer in enumerate(network.layers):
            vW_prev = self.vW[i].copy()
            vb_prev = self.vb[i].copy()
            self.vW[i] = self.momentum * self.vW[i] + self.learning_rate * layer.dW
            self.vb[i] = self.momentum * self.vb[i] + self.learning_rate * layer.db
            layer.W -= (1.0 + self.momentum) * self.vW[i] - self.momentum * vW_prev
            layer.b -= (1.0 + self.momentum) * self.vb[i] - self.momentum * vb_prev


class RMSprop(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.eps = eps
        self.sW = []
        self.sb = []

    def step(self, network: Network) -> None:
        if not self.sW:
            for l in network.layers:
                self.sW.append(np.zeros_like(l.W))
                self.sb.append(np.zeros_like(l.b))
        for i, layer in enumerate(network.layers):
            self.sW[i] = self.beta * self.sW[i] + (1.0 - self.beta) * (layer.dW ** 2)
            self.sb[i] = self.beta * self.sb[i] + (1.0 - self.beta) * (layer.db ** 2)
            layer.W -= self.learning_rate * layer.dW / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.learning_rate * layer.db / (np.sqrt(self.sb[i]) + self.eps)


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = []
        self.mb = []
        self.vW = []
        self.vb = []
        self.t = 0

    def step(self, network: Network) -> None:
        if not self.mW:
            for l in network.layers:
                self.mW.append(np.zeros_like(l.W))
                self.mb.append(np.zeros_like(l.b))
                self.vW.append(np.zeros_like(l.W))
                self.vb.append(np.zeros_like(l.b))
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        for i, layer in enumerate(network.layers):
            self.mW[i] = self.beta1 * self.mW[i] + (1.0 - self.beta1) * layer.dW
            self.mb[i] = self.beta1 * self.mb[i] + (1.0 - self.beta1) * layer.db
            self.vW[i] = self.beta2 * self.vW[i] + (1.0 - self.beta2) * (layer.dW ** 2)
            self.vb[i] = self.beta2 * self.vb[i] + (1.0 - self.beta2) * (layer.db ** 2)
            layer.W -= lr_t * self.mW[i] / (np.sqrt(self.vW[i]) + self.eps)
            layer.b -= lr_t * self.mb[i] / (np.sqrt(self.vb[i]) + self.eps)


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop": RMSprop,
    "adam": Adam,
}


def get_optimizer(name_or_instance, learning_rate: float = 0.01, **kwargs) -> Optimizer:
    if isinstance(name_or_instance, Optimizer):
        return name_or_instance
    if isinstance(name_or_instance, str):
        key = name_or_instance.lower()
        if key not in OPTIMIZERS:
            valid = ", ".join(OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer '{name_or_instance}'. Valid options: {valid}")
        return OPTIMIZERS[key](learning_rate=learning_rate, **kwargs)
    raise TypeError(f"Expected str or Optimizer, got {type(name_or_instance).__name__}")