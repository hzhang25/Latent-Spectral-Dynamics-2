from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from sklearn.decomposition import PCA

from .autoencoder import TrainedAutoencoder, train_autoencoder
from .config import AutoencoderConfig


class BasisMap(Protocol):
    def fit(self, x: np.ndarray) -> "BasisMap":
        ...

    def transform(self, x: np.ndarray, logits: Optional[np.ndarray] = None) -> np.ndarray:
        ...


@dataclass
class IdentityBasis:
    def fit(self, x: np.ndarray) -> "IdentityBasis":
        return self

    def transform(self, x: np.ndarray, logits: Optional[np.ndarray] = None) -> np.ndarray:
        return x


@dataclass
class TopKLogitsBasis:
    k: int

    def fit(self, x: np.ndarray) -> "TopKLogitsBasis":
        return self

    def transform(self, x: np.ndarray, logits: Optional[np.ndarray] = None) -> np.ndarray:
        if logits is None:
            raise ValueError("TopKLogitsBasis requires logits.")
        if logits.ndim != 3:
            raise ValueError("Expected logits shape [num_steps, T0, vocab].")
        token_mean_logits = logits.mean(axis=1)
        k = min(self.k, token_mean_logits.shape[-1])
        topk_idx = np.argpartition(token_mean_logits, kth=-k, axis=1)[:, -k:]
        topk_values = np.take_along_axis(token_mean_logits, topk_idx, axis=1)
        return np.sort(topk_values, axis=1)[:, ::-1]


@dataclass
class FourierBasis:
    low_freq_count: int

    def fit(self, x: np.ndarray) -> "FourierBasis":
        return self

    def transform(self, x: np.ndarray, logits: Optional[np.ndarray] = None) -> np.ndarray:
        spectra = np.fft.rfft(x, axis=1)
        use = min(self.low_freq_count, spectra.shape[1])
        low = spectra[:, :use]
        return np.concatenate([low.real, low.imag], axis=1)


@dataclass
class PCABasis:
    dim: int
    seed: int = 42
    model: Optional[PCA] = None

    def fit(self, x: np.ndarray) -> "PCABasis":
        dim = min(self.dim, x.shape[1], x.shape[0])
        self.model = PCA(n_components=dim, random_state=self.seed, svd_solver="randomized")
        self.model.fit(x)
        return self

    def transform(self, x: np.ndarray, logits: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        return self.model.transform(x)


@dataclass
class AutoencoderBasis:
    cfg: AutoencoderConfig
    trained: Optional[TrainedAutoencoder] = None

    def fit(self, x: np.ndarray) -> "AutoencoderBasis":
        self.trained = train_autoencoder(x, self.cfg)
        return self

    def transform(self, x: np.ndarray, logits: Optional[np.ndarray] = None) -> np.ndarray:
        if self.trained is None:
            raise RuntimeError("Call fit() before transform().")
        return self.trained.encode(x)

