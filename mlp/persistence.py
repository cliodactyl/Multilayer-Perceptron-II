"""Serialization and loading of trained model weights, topology, and scaling parameters."""

from typing import Tuple
import numpy as np

from mlp.dataset import Standardiser
from mlp.layers import DenseLayer
from mlp.network import Network


def save_model(network: Network, standardiser: Standardiser, path: str) -> None:
    architecture = []
    weights = []
    for layer in network.layers:
        architecture.append({
            "units": layer.units,
            "activation": layer.activation_name,
            "input_dim": layer.input_dim,
        })
        weights.append({
            "W": layer.W,
            "b": layer.b,
        })
    payload = {
        "architecture": architecture,
        "weights": weights,
        "standardiser": standardiser.to_dict(),
    }
    np.save(path, payload, allow_pickle=True)


def load_model(path: str) -> Tuple[Network, Standardiser]:
    raw = np.load(path, allow_pickle=True)
    payload = raw.item() if hasattr(raw, "item") else raw

    architecture = payload["architecture"]
    weights = payload["weights"]
    standardiser = Standardiser.from_dict(payload["standardiser"])

    network = Network()
    for layer_spec, w in zip(architecture, weights):
        layer = DenseLayer(
            units=layer_spec["units"],
            activation=layer_spec["activation"],
            input_dim=layer_spec["input_dim"],
        )
        layer.set_params(w["W"], w["b"])
        network.layers.append(layer)

    return network, standardiser