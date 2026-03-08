from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.sparse.linalg import eigs as sparse_eigs
except Exception:  # pragma: no cover - optional dependency fallback
    sparse_eigs = None

from experiment.config import ExperimentConfig


VALID_BASES = ["coordinate", "pca", "fourier"]
VALID_LAYERS = ["final", "middle", "all"]
LABEL_NAMES = {1: "truthful", 0: "untruthful"}


@dataclass(frozen=True)
class SequenceRecord:
    prompt_id: str
    label: int
    temperature: float
    rollout_id: int | None
    activation_sequence: np.ndarray  # [prompt_tail + generation_steps, feature_dim]


@dataclass(frozen=True)
class ObservableConfig:
    basis_name: str
    feature_dim: int
    operator_dim: int


@dataclass(frozen=True)
class FitResult:
    best_alpha: float
    train_mse: float
    test_mse: float
    operator_dim: int
    eigvals: np.ndarray
    fit_method: str


def _setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("lsd")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log: logging.Logger = logging.getLogger("lsd")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _complex_to_dict(value: complex) -> Dict[str, float]:
    return {
        "real": float(np.real(value)),
        "imag": float(np.imag(value)),
        "magnitude": float(np.abs(value)),
        "angle": float(np.angle(value)),
    }


def _top_mode_coordinates(vector: np.ndarray, top_n: int = 20) -> List[Dict[str, float]]:
    order = np.argsort(np.abs(vector))[::-1][:top_n]
    return [
        {
            "index": int(idx),
            "real": float(np.real(vector[idx])),
            "imag": float(np.imag(vector[idx])),
            "magnitude": float(np.abs(vector[idx])),
        }
        for idx in order
    ]


def _format_temperature_suffix(temperature: float) -> str:
    if float(temperature).is_integer():
        return str(int(temperature))
    return format(float(temperature), "g")


def _format_alpha_suffix(fixed_alpha: float | None) -> str:
    if fixed_alpha is None:
        return "grid"
    alpha_text = format(float(fixed_alpha), "g").replace("+", "")
    return f"alpha{_safe_name(alpha_text)}"


def _parse_label(raw_label: np.ndarray) -> int:
    if raw_label.dtype.kind in {"i", "u", "f"}:
        return int(raw_label.item())
    text = str(raw_label.item()).strip().lower()
    if text in {"truthful", "true", "1"}:
        return 1
    if text in {"untruthful", "false", "0"}:
        return 0
    raise ValueError(f"Unsupported label value: {text}")


def _resolve_temperature_dir(base_root: Path, deterministic_temp: float, temperature: float) -> Path:
    match = re.match(r"^(?P<prefix>.+?)_(?P<temp>\d+(?:\.\d+)?)(?P<suffix>(?:_.+)?)$", base_root.name)
    if match is not None:
        prefix = match.group("prefix")
        suffix = match.group("suffix")
        return base_root.with_name(f"{prefix}_{_format_temperature_suffix(temperature)}{suffix}")
    if np.isclose(float(temperature), float(deterministic_temp)):
        return base_root
    return base_root.with_name(f"{base_root.name}_{_format_temperature_suffix(temperature)}")


def _resolve_temperature_output_dir(
    base_output_dir: Path,
    temperature: float,
    max_pairs: int | None,
    layer_choice: str,
    fixed_alpha: float | None,
) -> Path:
    pair_suffix = "all" if max_pairs is None else str(int(max_pairs))
    return base_output_dir.with_name(
        f"{base_output_dir.name}_{_format_temperature_suffix(temperature)}_{_safe_name(layer_choice)}_{_format_alpha_suffix(fixed_alpha)}_{pair_suffix}"
    )


def _resolve_layer_index(layer_choice: str, num_layers: int) -> int:
    if layer_choice == "final":
        return num_layers - 1
    if layer_choice == "middle":
        return max(0, num_layers // 2 - 1)
    raise ValueError(f"Unsupported layer choice: {layer_choice}")


def _select_prompt_sequence(prompt_hl: np.ndarray, layer_choice: str, prompt_tail: int) -> np.ndarray:
    if prompt_hl.ndim != 3:
        raise ValueError(f"expected prompt_hl to have shape [L, prompt_len, d], got {prompt_hl.shape}")
    if prompt_hl.shape[1] < prompt_tail:
        raise ValueError(f"prompt has only {prompt_hl.shape[1]} tokens, fewer than prompt_tail={prompt_tail}.")
    if layer_choice == "all":
        # Concatenate all layer activations for each prompt token.
        return (
            np.transpose(prompt_hl[:, -prompt_tail:, :], (1, 0, 2))
            .reshape(prompt_tail, -1)
            .astype(np.float32, copy=False)
        )
    layer_idx = _resolve_layer_index(layer_choice, prompt_hl.shape[0])
    return prompt_hl[layer_idx, -prompt_tail:, :].astype(np.float32, copy=False)


def _select_generated_sequence(hk: np.ndarray, layer_choice: str, generation_steps: int) -> np.ndarray:
    if hk.ndim != 4:
        raise ValueError(f"expected hk to have shape [num_blocks, L, T0, d], got {hk.shape}")
    if layer_choice == "all":
        # Concatenate all layer activations for each generated token.
        generated_seq = np.transpose(hk, (0, 2, 1, 3)).reshape(-1, hk.shape[1] * hk.shape[3])
    else:
        layer_idx = _resolve_layer_index(layer_choice, hk.shape[1])
        generated_seq = hk[:, layer_idx, :, :].reshape(-1, hk.shape[-1])
    generated_seq = generated_seq.astype(np.float32, copy=False)
    if generated_seq.shape[0] < generation_steps:
        raise ValueError(
            f"only {generated_seq.shape[0]} generated tokens available, need {generation_steps}."
        )
    return generated_seq[:generation_steps]


def _pair_number_from_filename(path: Path) -> int | None:
    match = re.match(r"^pair_(\d+)_", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _extract_sequence_record(
    npz,
    layer_choice: str,
    prompt_tail: int,
    generation_steps: int,
) -> SequenceRecord:
    prompt_id = str(npz["prompt_id"].item())
    label = _parse_label(npz["label"])
    temperature = float(npz["temperature"].item())
    rollout_id = int(npz["rollout_id"].item()) if "rollout_id" in npz else None

    prompt_hl = np.asarray(npz["prompt_hl"])
    if prompt_hl.ndim != 3:
        raise ValueError(f"{prompt_id}: expected prompt_hl to have shape [L, prompt_len, d].")
    prompt_seq = _select_prompt_sequence(prompt_hl, layer_choice=layer_choice, prompt_tail=prompt_tail)

    hk = np.asarray(npz["hk"])
    if hk.ndim != 4:
        raise ValueError(f"{prompt_id}: expected hk to have shape [num_blocks, L, T0, d].")
    generated_seq = _select_generated_sequence(hk, layer_choice=layer_choice, generation_steps=generation_steps)
    activation_sequence = np.concatenate([prompt_seq, generated_seq], axis=0)

    return SequenceRecord(
        prompt_id=prompt_id,
        label=label,
        temperature=temperature,
        rollout_id=rollout_id,
        activation_sequence=activation_sequence,
    )


def _load_temperature_records(
    data_dir: Path,
    temperature: float,
    layer_choice: str,
    prompt_tail: int,
    generation_steps: int,
    max_pairs: int | None,
) -> List[SequenceRecord]:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    chosen: Dict[str, SequenceRecord] = {}
    duplicates = 0
    for file in files:
        pair_number_from_name = _pair_number_from_filename(file)
        if max_pairs is not None and pair_number_from_name is not None and pair_number_from_name > max_pairs:
            # Files are sorted lexicographically and prompt ids are zero-padded,
            # so once we pass the requested pair cap we can stop without loading
            # the remaining large .npz archives.
            break
        with np.load(file, allow_pickle=True) as npz:
            file_temp = float(npz["temperature"].item())
            if not np.isclose(file_temp, temperature):
                continue
            record = _extract_sequence_record(
                npz=npz,
                layer_choice=layer_choice,
                prompt_tail=prompt_tail,
                generation_steps=generation_steps,
            )
        if max_pairs is not None:
            pair_number = pair_number_from_name
            if pair_number is None:
                pair_number = int(record.prompt_id.split("_")[1])
            if pair_number > max_pairs:
                continue
        existing = chosen.get(record.prompt_id)
        if existing is None:
            chosen[record.prompt_id] = record
            continue
        duplicates += 1
        existing_rollout = existing.rollout_id if existing.rollout_id is not None else -1
        record_rollout = record.rollout_id if record.rollout_id is not None else -1
        if record_rollout < existing_rollout:
            chosen[record.prompt_id] = record

    records = list(sorted(chosen.values(), key=lambda rec: rec.prompt_id))
    if duplicates:
        log.warning(
            "Temperature %.1f: found %d duplicate prompt records; kept the lowest rollout id per prompt.",
            temperature,
            duplicates,
        )
    return records


def _split_train_test_promptwise(
    records: Sequence[SequenceRecord],
    train_fraction: float,
    seed: int,
) -> tuple[List[SequenceRecord], List[SequenceRecord]]:
    if not records:
        return [], []
    prompt_ids = np.array([record.prompt_id for record in records])
    order = np.argsort(prompt_ids)
    ordered = [records[idx] for idx in order]
    rng = np.random.default_rng(seed)
    shuffled_indices = np.arange(len(ordered))
    rng.shuffle(shuffled_indices)
    split_idx = max(1, int(round(train_fraction * len(ordered))))
    split_idx = min(split_idx, len(ordered) - 1)
    train = [ordered[idx] for idx in shuffled_indices[:split_idx]]
    test = [ordered[idx] for idx in shuffled_indices[split_idx:]]
    return train, test


def _fit_pca_from_training_records(
    records: Sequence[SequenceRecord],
    pca_dim: int,
    pca_sample_size: int,
    seed: int,
) -> PCA:
    all_activations = np.concatenate([record.activation_sequence for record in records], axis=0)
    sample_size = min(pca_sample_size, all_activations.shape[0])
    if sample_size < 2:
        raise RuntimeError("Need at least two activations to fit PCA.")
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(all_activations.shape[0], size=sample_size, replace=False)
    sampled = all_activations[sample_idx]
    dim = min(pca_dim, sampled.shape[0], sampled.shape[1])
    model = PCA(n_components=dim, random_state=seed, svd_solver="randomized")
    model.fit(sampled)
    return model


def _transform_sequence(
    record: SequenceRecord,
    basis_name: str,
    pca_model: PCA | None,
    fourier_low_freq_count: int,
) -> np.ndarray:
    if basis_name == "coordinate":
        return record.activation_sequence.astype(np.float32, copy=False)
    if basis_name == "pca":
        if pca_model is None:
            raise RuntimeError("PCA basis requested without a fitted PCA model.")
        transformed = pca_model.transform(record.activation_sequence.astype(np.float32, copy=False))
        return transformed.astype(np.float32, copy=False)
    if basis_name == "fourier":
        spectra = np.fft.rfft(record.activation_sequence.astype(np.float32, copy=False), axis=1)
        use = min(fourier_low_freq_count, spectra.shape[1])
        low = spectra[:, :use]
        transformed = np.concatenate([low.real, low.imag], axis=1)
        return transformed.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported basis: {basis_name}")


def _make_hankel_transitions(sequence: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence.shape[0] < window + 1:
        raise ValueError(
            f"Need at least window+1={window + 1} steps to form one-step Hankel transitions; "
            f"got {sequence.shape[0]}."
        )
    hankel = np.stack(
        [sequence[idx : idx + window].reshape(-1) for idx in range(sequence.shape[0] - window + 1)],
        axis=0,
    ).astype(np.float32, copy=False)
    return hankel[:-1], hankel[1:]


def _build_prompt_transition_matrices(
    records: Sequence[SequenceRecord],
    basis_name: str,
    hankel_window: int,
    pca_model: PCA | None,
    fourier_low_freq_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    feature_dim = 0
    for record in records:
        sequence = _transform_sequence(
            record,
            basis_name=basis_name,
            pca_model=pca_model,
            fourier_low_freq_count=fourier_low_freq_count,
        )
        feature_dim = int(sequence.shape[1])
        x_prompt, y_prompt = _make_hankel_transitions(sequence, window=hankel_window)
        x_rows.append(x_prompt)
        y_rows.append(y_prompt)
    x = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)
    return x, y, feature_dim


def _fit_direct_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alphas: Sequence[float],
) -> tuple[FitResult, np.ndarray]:
    best_result: FitResult | None = None
    best_operator: np.ndarray | None = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_mse = float(np.mean((train_pred - y_train) ** 2))
        test_mse = float(np.mean((test_pred - y_test) ** 2))
        operator = model.coef_.astype(np.float64, copy=False)
        if best_result is None or test_mse < best_result.test_mse:
            eigvals = _compute_eigvals(operator, top_n=min(128, operator.shape[0]))
            best_result = FitResult(
                best_alpha=float(alpha),
                train_mse=train_mse,
                test_mse=test_mse,
                operator_dim=int(operator.shape[0]),
                eigvals=eigvals,
                fit_method="ridge_direct",
            )
            best_operator = operator
    if best_result is None or best_operator is None:
        raise RuntimeError("Ridge fit failed to produce a valid result.")
    return best_result, best_operator


def _chunked_prediction_mse(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    chunk_cols: int = 1024,
) -> float:
    if target.size == 0:
        return 0.0
    total_sse = 0.0
    for start in range(0, target.shape[1], chunk_cols):
        stop = min(start + chunk_cols, target.shape[1])
        pred_chunk = left @ right[:, start:stop]
        diff = pred_chunk - target[:, start:stop]
        total_sse += float(np.sum(diff * diff))
    return total_sse / float(target.shape[0] * target.shape[1])


def _compute_leading_eigendecomposition(
    matrix: np.ndarray,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    if n <= 2048:
        eigvals, eigvecs = np.linalg.eig(matrix)
    else:
        if sparse_eigs is None:
            log.warning("scipy.sparse.linalg.eigs unavailable; skipping large-matrix eigendecomposition.")
            return np.asarray([], dtype=np.complex128), np.zeros((n, 0), dtype=np.complex128)
        k = max(1, min(top_n, n - 2))
        eigvals, eigvecs = sparse_eigs(matrix, k=k, which="LM", return_eigenvectors=True)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[order[:top_n]]
    eigvecs = eigvecs[:, order[:top_n]]
    return eigvals, eigvecs


def _fit_coordinate_dual_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alphas: Sequence[float],
) -> tuple[FitResult, np.ndarray, np.ndarray]:
    x_train64 = x_train.astype(np.float64, copy=False)
    y_train64 = y_train.astype(np.float64, copy=False)
    x_test64 = x_test.astype(np.float64, copy=False)
    y_test64 = y_test.astype(np.float64, copy=False)

    gram_train = x_train64 @ x_train64.T
    cross_test = x_test64 @ x_train64.T
    small_rhs = y_train64 @ x_train64.T
    identity = np.eye(gram_train.shape[0], dtype=np.float64)
    best_alpha: float | None = None
    best_train_mse: float | None = None
    best_test_mse: float | None = None

    for alpha in alphas:
        alpha = float(alpha)
        solve_matrix = gram_train + alpha * identity
        dual_weights = np.linalg.solve(solve_matrix, y_train64)
        # Since (G + aI)A = Y, the training residual is G A - Y = -a A.
        train_mse = float((alpha * alpha) * np.mean(dual_weights * dual_weights))
        test_mse = _chunked_prediction_mse(cross_test, dual_weights, y_test64)
        # Dual/kernel DMD operator with the same non-zero spectrum as the
        # primal coordinate-basis ridge operator K = (X^T X + aI)^-1 X^T Y.
        # This orientation is chosen so primal activation-space modes lift as X^T v_dual.
        if best_test_mse is None or test_mse < best_test_mse:
            best_alpha = alpha
            best_train_mse = train_mse
            best_test_mse = test_mse
    if best_alpha is None or best_train_mse is None or best_test_mse is None:
        raise RuntimeError("Dual ridge fit failed to produce a valid result.")
    best_solve_matrix = gram_train + best_alpha * identity
    best_small_operator = np.linalg.solve(best_solve_matrix, small_rhs)
    eigvals, best_dual_eigvecs = _compute_leading_eigendecomposition(
        best_small_operator,
        top_n=min(128, best_small_operator.shape[0]),
    )
    best_result = FitResult(
        best_alpha=best_alpha,
        train_mse=best_train_mse,
        test_mse=best_test_mse,
        operator_dim=int(x_train.shape[1]),
        eigvals=eigvals,
        fit_method="ridge_dual_coordinate",
    )
    return best_result, best_small_operator, best_dual_eigvecs


def _compute_eigvals(matrix: np.ndarray, top_n: int) -> np.ndarray:
    eigvals, _ = _compute_leading_eigendecomposition(matrix, top_n=top_n)
    return eigvals


def _plot_eigenvalues(eigvals: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", color="gray", linewidth=1, label="unit circle")
    if eigvals.size:
        ax.scatter(eigvals.real, eigvals.imag, s=20, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_truth_lie_overlay(
    truth_eigvals: np.ndarray,
    lie_eigvals: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", color="gray", linewidth=1, label="unit circle")
    if truth_eigvals.size:
        ax.scatter(truth_eigvals.real, truth_eigvals.imag, s=18, alpha=0.75, label="truth")
    if lie_eigvals.size:
        ax.scatter(lie_eigvals.real, lie_eigvals.imag, s=18, alpha=0.75, label="lie")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_mse_summary(results: List[Dict[str, object]], out_path: Path) -> None:
    if not results:
        return
    ordered = sorted(results, key=lambda item: (item["temperature"], item["basis"], item["label_name"]))
    labels = [
        f"tau={item['temperature']}\n{item['basis']}\n{item['label_name']}"
        for item in ordered
    ]
    mse_values = [float(item["test_mse"]) for item in ordered]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.9), 5))
    ax.bar(x, mse_values)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Held-out 1-step MSE")
    ax.set_title("Held-out EDMD prediction error by temperature / basis / label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_combo(
    temperature: float,
    label: int,
    records: Sequence[SequenceRecord],
    basis_name: str,
    layer_choice: str,
    hankel_window: int,
    train_fraction: float,
    ridge_alphas: Sequence[float],
    fixed_alpha: float | None,
    pca_dim: int,
    pca_sample_size: int,
    fourier_low_freq_count: int,
    seed: int,
    artifact_dir: Path,
) -> Dict[str, object]:
    label_name = LABEL_NAMES[label]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    train_records, test_records = _split_train_test_promptwise(records, train_fraction=train_fraction, seed=seed)
    if not train_records or not test_records:
        raise RuntimeError(f"Need non-empty train and test splits for tau={temperature}, label={label_name}.")

    pca_model: PCA | None = None
    if basis_name == "pca":
        pca_model = _fit_pca_from_training_records(
            train_records,
            pca_dim=pca_dim,
            pca_sample_size=pca_sample_size,
            seed=seed,
        )

    x_train, y_train, feature_dim = _build_prompt_transition_matrices(
        train_records,
        basis_name=basis_name,
        hankel_window=hankel_window,
        pca_model=pca_model,
        fourier_low_freq_count=fourier_low_freq_count,
    )
    x_test, y_test, _ = _build_prompt_transition_matrices(
        test_records,
        basis_name=basis_name,
        hankel_window=hankel_window,
        pca_model=pca_model,
        fourier_low_freq_count=fourier_low_freq_count,
    )
    operator_dim = feature_dim * hankel_window
    observable_cfg = ObservableConfig(
        basis_name=basis_name,
        feature_dim=feature_dim,
        operator_dim=operator_dim,
    )

    if basis_name == "coordinate":
        fit_result, saved_operator, dual_eigvecs = _fit_coordinate_dual_ridge(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            alphas=ridge_alphas,
        )
        primal_modes = x_train.T.astype(np.float64, copy=False) @ dual_eigvecs
        if primal_modes.size:
            mode_norms = np.linalg.norm(primal_modes, axis=0, keepdims=True)
            mode_norms[mode_norms < 1e-12] = 1.0
            primal_modes = primal_modes / mode_norms
        modes_json_path = artifact_dir / "coordinate_modes.json"
        _save_json(
            modes_json_path,
            {
                "mode_lifting_formula": "v_primal = X_train^T v_dual",
                "dual_operator_formula": "(X X^T + alpha I)^-1 (Y X^T)",
                "mode_count": int(primal_modes.shape[1]),
                "top_modes": [
                    {
                        "rank": mode_idx + 1,
                        "eigenvalue": _complex_to_dict(fit_result.eigvals[mode_idx]),
                        "top_coordinates": _top_mode_coordinates(primal_modes[:, mode_idx], top_n=20),
                    }
                    for mode_idx in range(min(8, primal_modes.shape[1]))
                ],
            },
        )
        operator_save_path = artifact_dir / "koopman_dual_operator.npz"
        np.savez_compressed(
            operator_save_path,
            reduced_operator=saved_operator,
            eigvals=fit_result.eigvals,
            dual_eigvecs=dual_eigvecs,
            primal_modes=primal_modes,
            note=np.array(
                "Coordinate basis uses a dual ridge operator; non-zero primal eigenvalues match the saved dual eigenvalues, and primal modes are lifted as X_train^T @ dual_eigvecs."
            ),
        )
    else:
        fit_result, saved_operator = _fit_direct_ridge(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            alphas=ridge_alphas,
        )
        operator_save_path = artifact_dir / "koopman_operator.npz"
        np.savez_compressed(operator_save_path, koopman_matrix=saved_operator, eigvals=fit_result.eigvals)

    _plot_eigenvalues(
        fit_result.eigvals,
        artifact_dir / "eigenvalues.png",
        title=f"tau={temperature} / {label_name} / {basis_name} / layer={layer_choice}",
    )
    elapsed = time.perf_counter() - t_start

    payload = {
        "temperature": float(temperature),
        "label": int(label),
        "label_name": label_name,
        "layer": layer_choice,
        "basis": basis_name,
        "prompt_count_total": int(len(records)),
        "prompt_count_train": int(len(train_records)),
        "prompt_count_test": int(len(test_records)),
        "train_transitions": int(x_train.shape[0]),
        "test_transitions": int(x_test.shape[0]),
        "feature_dim": int(observable_cfg.feature_dim),
        "operator_dim": int(observable_cfg.operator_dim),
        "best_alpha": float(fit_result.best_alpha),
        "alpha_mode": "fixed" if fixed_alpha is not None else "grid_search",
        "alpha_candidates": [float(alpha) for alpha in ridge_alphas],
        "train_mse": float(fit_result.train_mse),
        "test_mse": float(fit_result.test_mse),
        "fit_method": fit_result.fit_method,
        "top_eigenvalues": [_complex_to_dict(value) for value in fit_result.eigvals[:32]],
        "artifacts": {
            "operator_npz": str(operator_save_path),
            "eigenvalues_plot": str(artifact_dir / "eigenvalues.png"),
        },
        "elapsed_seconds": float(elapsed),
    }
    if basis_name == "coordinate":
        payload["num_primal_modes_saved"] = int(dual_eigvecs.shape[1])
        payload["artifacts"]["coordinate_modes_json"] = str(modes_json_path)
    if basis_name == "pca" and pca_model is not None:
        payload["pca_components_used"] = int(pca_model.n_components_)
        payload["pca_sample_size"] = int(
            min(pca_sample_size, sum(record.activation_sequence.shape[0] for record in train_records))
        )
    if basis_name == "fourier":
        payload["fourier_low_freq_count"] = int(fourier_low_freq_count)
    log.info(
        "DONE  tau=%.1f  label=%-11s  basis=%-10s  layer=%-6s  train_mse=%.4e  test_mse=%.4e  alpha=%g  time=%.1fs",
        temperature,
        label_name,
        basis_name,
        layer_choice,
        fit_result.train_mse,
        fit_result.test_mse,
        fit_result.best_alpha,
        elapsed,
    )
    return payload


def _plot_overlay_summaries(results: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key: Dict[tuple[float, str], Dict[str, np.ndarray]] = {}
    for result in results:
        key = (float(result["temperature"]), str(result["basis"]))
        label_name = str(result["label_name"])
        eigvals = np.asarray(
            [
                complex(item["real"], item["imag"])
                for item in result["top_eigenvalues"]
            ],
            dtype=np.complex128,
        )
        by_key.setdefault(key, {})[label_name] = eigvals

    for (temperature, basis_name), item in by_key.items():
        if "truthful" not in item or "untruthful" not in item:
            continue
        out_path = out_dir / f"overlay_tau_{_format_temperature_suffix(temperature)}_{_safe_name(basis_name)}.png"
        _plot_truth_lie_overlay(
            truth_eigvals=item["truthful"],
            lie_eigvals=item["untruthful"],
            out_path=out_path,
            title=f"tau={temperature} / basis={basis_name}: truth vs lie",
        )


def run_full_experiment(
    data_root: str,
    output_dir: str,
    layer_choice: str,
    temperatures: Sequence[float],
    max_pairs: Optional[int] = None,
    bases: Optional[List[str]] = None,
    fixed_alpha: float | None = None,
    hankel_window: int = 5,
    prompt_tail: int = 5,
    generation_steps: int = 32,
    pca_dim: int = 256,
    pca_sample_size: int = 10_000,
    fourier_low_freq_count: int = 32,
) -> None:
    t_total = time.perf_counter()
    cfg = ExperimentConfig()
    effective_bases = bases if bases is not None else list(VALID_BASES)
    ridge_alphas = [float(fixed_alpha)] if fixed_alpha is not None else list(cfg.probe.regularization_grid)
    data_root_path = Path(data_root)
    base_out_dir = Path(output_dir)

    log.info("=" * 72)
    log.info("Token-level Koopman experiment")
    log.info("  model              : %s", cfg.model_name)
    log.info("  data root          : %s", data_root_path.resolve())
    log.info("  output dir base    : %s", base_out_dir.resolve())
    log.info("  layer choice       : %s", layer_choice)
    log.info("  temperatures       : %s", [float(temp) for temp in temperatures])
    log.info("  max pairs          : %s", max_pairs if max_pairs is not None else "all")
    log.info("  prompt tail tokens : %d", prompt_tail)
    log.info("  generated steps    : %d", generation_steps)
    log.info("  hankel window      : %d", hankel_window)
    log.info("  bases              : %s", effective_bases)
    log.info("  train fraction     : %.0f%%", cfg.probe.train_fraction * 100)
    if fixed_alpha is not None:
        log.info("  fixed alpha        : %g", float(fixed_alpha))
    else:
        log.info("  ridge grid         : %s", ridge_alphas)
    log.info("  PCA dim            : %d", pca_dim)
    log.info("  PCA sample size    : %d", pca_sample_size)
    log.info("  Fourier low-freq   : %d", fourier_low_freq_count)
    log.info("=" * 72)

    saved_outputs: List[str] = []

    for temperature in temperatures:
        temp_out_dir = _resolve_temperature_output_dir(
            base_output_dir=base_out_dir,
            temperature=float(temperature),
            max_pairs=max_pairs,
            layer_choice=layer_choice,
            fixed_alpha=fixed_alpha,
        )
        temp_out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output for tau=%.1f will be written to %s", temperature, temp_out_dir.resolve())
        temp_dir = _resolve_temperature_dir(
            base_root=data_root_path,
            deterministic_temp=cfg.deterministic_temperature,
            temperature=float(temperature),
        )
        if not temp_dir.exists():
            log.warning("Temperature directory missing for tau=%.1f: %s", temperature, temp_dir)
            continue
        log.info("Loading tau=%.1f records from %s", temperature, temp_dir)
        records = _load_temperature_records(
            data_dir=temp_dir,
            temperature=float(temperature),
            layer_choice=layer_choice,
            prompt_tail=prompt_tail,
            generation_steps=generation_steps,
            max_pairs=max_pairs,
        )
        if not records:
            log.warning("No usable records for tau=%.1f", temperature)
            continue

        temp_results: List[Dict[str, object]] = []
        artifacts_root = temp_out_dir / "artifacts"
        by_label: Dict[int, List[SequenceRecord]] = {0: [], 1: []}
        for record in records:
            by_label[record.label].append(record)
        for label, label_records in by_label.items():
            if not label_records:
                log.warning("No %s records found for tau=%.1f", LABEL_NAMES[label], temperature)
                continue
            log.info(
                "tau=%.1f  label=%s  prompts=%d",
                temperature,
                LABEL_NAMES[label],
                len(label_records),
            )
            for basis_name in effective_bases:
                combo_dir = (
                    artifacts_root
                    / f"tau_{_format_temperature_suffix(temperature)}"
                    / LABEL_NAMES[label]
                    / _safe_name(layer_choice)
                    / _safe_name(basis_name)
                )
                combo_result = _run_combo(
                    temperature=float(temperature),
                    label=label,
                    records=label_records,
                    basis_name=basis_name,
                    layer_choice=layer_choice,
                    hankel_window=hankel_window,
                    train_fraction=cfg.probe.train_fraction,
                    ridge_alphas=ridge_alphas,
                    fixed_alpha=fixed_alpha,
                    pca_dim=pca_dim,
                    pca_sample_size=pca_sample_size,
                    fourier_low_freq_count=fourier_low_freq_count,
                    seed=cfg.probe.random_seed,
                    artifact_dir=combo_dir,
                )
                temp_results.append(combo_result)

        _save_json(temp_out_dir / "experiment_results.json", {"results": temp_results})
        _plot_mse_summary(temp_results, temp_out_dir / "heldout_mse_summary.png")
        _plot_overlay_summaries(temp_results, temp_out_dir / "comparisons")
        saved_outputs.append(str(temp_out_dir.resolve()))

    elapsed_total = time.perf_counter() - t_total
    log.info("=" * 72)
    log.info(
        "Experiment complete. temperatures=%d  elapsed=%.1fs (%.1f min)",
        len(saved_outputs),
        elapsed_total,
        elapsed_total / 60,
    )
    for saved_output in saved_outputs:
        log.info("Results saved to: %s", saved_output)
    log.info("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run token-level Koopman EDMD experiments over extracted Qwen activations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="data_0",
        help=(
            "Deterministic temperature directory. Sibling directories for other temperatures are inferred\n"
            "from this path, e.g. data_0 -> data_0.3 and data_0.7, or data_0_simple -> data_0.3_simple."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/token_level",
        help="Directory to write experiment JSON and plots.",
    )
    parser.add_argument(
        "--layer",
        choices=VALID_LAYERS,
        default="final",
        help="Hidden representation to analyze: final, middle, or all layers concatenated per token.",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.7],
        metavar="TAU",
        help="Temperatures to evaluate. Directories are inferred from --data-root.",
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        choices=VALID_BASES,
        default=list(VALID_BASES),
        metavar="BASIS",
        help="Observable bases to evaluate.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        metavar="N",
        help="Optional cap on prompt pairs. Uses at most one truthful and one untruthful prompt per pair.",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="Optional fixed ridge penalty. If set, skips alpha grid search and fits only this value.",
    )
    parser.add_argument(
        "--prompt-tail",
        type=int,
        default=5,
        help="Number of prompt tokens to keep before generation.",
    )
    parser.add_argument(
        "--generation-steps",
        type=int,
        default=32,
        help="Number of generated tokens to keep per prompt.",
    )
    parser.add_argument(
        "--hankel-window",
        type=int,
        default=5,
        help="Hankel window size W over token-level observables.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=256,
        help="PCA latent dimension before Hankel concatenation.",
    )
    parser.add_argument(
        "--pca-sample-size",
        type=int,
        default=10_000,
        help="Random activation sample count used to fit PCA on the training split.",
    )
    parser.add_argument(
        "--fourier-low-freq-count",
        type=int,
        default=32,
        help="Number of low-frequency Fourier coefficients to retain per token activation.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to mirror log output to a file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logger = _setup_logging(log_file=args.log_file)
    logger.setLevel(getattr(logging, args.log_level))

    run_full_experiment(
        data_root=args.data_root,
        output_dir=args.output_dir,
        layer_choice=args.layer,
        temperatures=args.temperatures,
        max_pairs=args.max_pairs,
        bases=args.bases,
        fixed_alpha=args.fixed_alpha,
        hankel_window=args.hankel_window,
        prompt_tail=args.prompt_tail,
        generation_steps=args.generation_steps,
        pca_dim=args.pca_dim,
        pca_sample_size=args.pca_sample_size,
        fourier_low_freq_count=args.fourier_low_freq_count,
    )


if __name__ == "__main__":
    main()
