from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .basis import BasisMap


@dataclass
class EDMDResult:
    koopman_matrix: np.ndarray
    eigvals: np.ndarray
    train_mse: float
    test_mse: Optional[float]


def make_hankel(states: np.ndarray, delay: int) -> np.ndarray:
    """
    states: [num_steps, state_dim]
    return: [num_steps - delay, (delay + 1) * state_dim]
    """
    if delay < 0:
        raise ValueError("Delay must be non-negative.")
    if states.shape[0] <= delay + 1:
        raise ValueError("Not enough steps for given delay.")

    chunks = []
    for idx in range(delay, states.shape[0]):
        block = [states[idx - d] for d in range(delay, -1, -1)]
        chunks.append(np.concatenate(block, axis=0))
    return np.asarray(chunks)


def one_step_pairs(hankel_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = hankel_states[:-1]
    y = hankel_states[1:]
    return x, y


def fit_edmd(
    x: np.ndarray,
    y: np.ndarray,
    basis: BasisMap,
    x_logits: Optional[np.ndarray] = None,
    y_logits: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
    test_fraction: float = 0.2,
) -> EDMDResult:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of transitions.")
    if x.shape[0] < 2:
        raise ValueError("Need at least two transitions for EDMD fitting.")

    split_idx = max(1, int((1.0 - test_fraction) * x.shape[0]))
    split_idx = min(split_idx, x.shape[0] - 1)

    x_train, y_train = x[:split_idx], y[:split_idx]
    x_test, y_test = x[split_idx:], y[split_idx:]
    x_logits_train = x_logits[:split_idx] if x_logits is not None else None
    y_logits_train = y_logits[:split_idx] if y_logits is not None else None
    x_logits_test = x_logits[split_idx:] if x_logits is not None else None
    y_logits_test = y_logits[split_idx:] if y_logits is not None else None

    basis.fit(x_train)
    phi_x = basis.transform(x_train, x_logits_train)
    phi_y = basis.transform(y_train, y_logits_train)

    xx_t = phi_x.T @ phi_x
    xy_t = phi_x.T @ phi_y
    reg = ridge * np.eye(xx_t.shape[0])
    koopman_t = np.linalg.solve(xx_t + reg, xy_t)
    koopman = koopman_t.T

    pred = phi_x @ koopman_t
    train_mse = float(np.mean((pred - phi_y) ** 2))

    test_mse: Optional[float]
    if x_test.shape[0] > 0:
        phi_x_test = basis.transform(x_test, x_logits_test)
        phi_y_test = basis.transform(y_test, y_logits_test)
        test_pred = phi_x_test @ koopman_t
        test_mse = float(np.mean((test_pred - phi_y_test) ** 2))
    else:
        test_mse = None
    eigvals = np.linalg.eigvals(koopman)
    return EDMDResult(koopman_matrix=koopman, eigvals=eigvals, train_mse=train_mse, test_mse=test_mse)


def spectral_features(eigvals: np.ndarray, top_n: int = 32) -> np.ndarray:
    order = np.argsort(np.abs(eigvals))[::-1]
    chosen = eigvals[order[:top_n]]
    if chosen.shape[0] < top_n:
        pad = np.zeros(top_n - chosen.shape[0], dtype=complex)
        chosen = np.concatenate([chosen, pad], axis=0)
    return np.concatenate([chosen.real, chosen.imag], axis=0)

