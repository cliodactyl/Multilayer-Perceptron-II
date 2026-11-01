"""Weight initialisation schemes (He and Glorot / Xavier distributions)."""

import numpy as np


class Initialiser:
    def __call__(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class HeUniform(Initialiser):
    def __call__(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        fan_in = shape[0]
        limit = np.sqrt(6.0 / fan_in)
        return rng.uniform(-limit, limit, size=shape)


class HeNormal(Initialiser):
    def __call__(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(0.0, std, size=shape)


class GlorotUniform(Initialiser):
    def __call__(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape)


class GlorotNormal(Initialiser):
    def __call__(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        fan_in, fan_out = shape[0], shape[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0.0, std, size=shape)


class Zeros(Initialiser):
    def __call__(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(shape, dtype=float)


INITIALISERS = {
    "heuniform": HeUniform,
    "henormal": HeNormal,
    "glorotuniform": GlorotUniform,
    "glorotnormal": GlorotNormal,
    "xavieruniform": GlorotUniform,
    "xaviernormal": GlorotNormal,
    "zeros": Zeros,
}


def get_initialiser(name_or_instance) -> Initialiser:
    if isinstance(name_or_instance, Initialiser):
        return name_or_instance
    if isinstance(name_or_instance, str):
        key = name_or_instance.lower()
        if key not in INITIALISERS:
            valid = ", ".join(INITIALISERS.keys())
            raise ValueError(f"Unrecognised initialiser '{name_or_instance}'. Valid options: {valid}")
        return INITIALISERS[key]()
    raise TypeError(f"Expected str or Initialiser, got {type(name_or_instance).__name__}")