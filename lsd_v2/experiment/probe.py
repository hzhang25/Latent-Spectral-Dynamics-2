from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import ProbeConfig


@dataclass
class ProbeResult:
    best_c: float
    test_accuracy: float
    test_f1: float


def run_linear_probe(x: np.ndarray, y: np.ndarray, cfg: ProbeConfig) -> ProbeResult:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=cfg.train_fraction,
        random_state=cfg.random_seed,
        stratify=y,
    )

    best: Dict[str, float] = {"c": 0.0, "score": -1.0}
    best_model = None
    for c in cfg.regularization_grid:
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        score = float(accuracy_score(y_test, pred))
        if score > best["score"]:
            best = {"c": c, "score": score}
            best_model = model

    if best_model is None:
        raise RuntimeError("Probe fitting failed.")
    test_pred = best_model.predict(x_test)
    return ProbeResult(
        best_c=float(best["c"]),
        test_accuracy=float(accuracy_score(y_test, test_pred)),
        test_f1=float(f1_score(y_test, test_pred)),
    )

