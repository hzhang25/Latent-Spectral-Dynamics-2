from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    matplotlib = None
    plt = None

from experiment.config import ExperimentConfig


log = logging.getLogger("lsd.multistep")
VALID_BASES = ["coordinate", "pca", "fourier"]
IGNORED_OUTPUT_DIR_NAMES = {"tier1", "last_token_pca"}
IGNORED_LAYER_CHOICES = {"all"}


@dataclass(frozen=True)
class ComboSpec:
    output_dir: Path
    results_path: Path
    payload: Dict[str, object]


@dataclass(frozen=True)
class RolloutMetric:
    horizon: int
    mse: float
    rmse: float
    relative_mse: float
    state_count: int
    scalar_count: int


@dataclass(frozen=True)
class SequenceRecord:
    prompt_id: str
    label: int
    temperature: float
    rollout_id: int | None
    activation_sequence: np.ndarray


@dataclass(frozen=True)
class PCAProjector:
    mean_: np.ndarray
    components_: np.ndarray

    @property
    def n_components_(self) -> int:
        return int(self.components_.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        centered = x.astype(np.float32, copy=False) - self.mean_
        return centered @ self.components_.T


def _setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("lsd.multistep")
    logger.setLevel(getattr(logging, log_level))
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level))
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(getattr(logging, log_level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _parse_pair_cap(output_dir: Path) -> int | None:
    tail = output_dir.name.rsplit("_", 1)[-1]
    if tail == "all":
        return None
    if tail.isdigit():
        return int(tail)
    return None


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _matches_allowed_temperature(value: object, allowed_temperatures: Sequence[float]) -> bool:
    try:
        temperature = float(value)
    except Exception:
        return False
    return any(np.isclose(temperature, float(candidate)) for candidate in allowed_temperatures)


def _infer_alpha_mode(item: Dict[str, object], output_dir: Path) -> str:
    raw_mode = item.get("alpha_mode")
    if raw_mode is not None:
        mode = str(raw_mode).strip()
        if mode:
            return mode
    # Older experiment outputs did not save alpha_mode explicitly.
    # Infer fixed-alpha runs from the directory name and treat the rest as grid search.
    parts = output_dir.name.split("_")
    if any(part.startswith("alpha") for part in parts):
        return "fixed"
    return "grid_search"


def _parse_label(raw_label: np.ndarray) -> int:
    if raw_label.dtype.kind in {"i", "u", "f"}:
        return int(raw_label.item())
    text = str(raw_label.item()).strip().lower()
    if text in {"truthful", "true", "1"}:
        return 1
    if text in {"untruthful", "false", "0"}:
        return 0
    raise ValueError(f"Unsupported label value: {text}")


def _format_temperature_suffix(temperature: float) -> str:
    if float(temperature).is_integer():
        return str(int(temperature))
    return format(float(temperature), "g")


def _resolve_temperature_dir(base_root: Path, deterministic_temp: float, temperature: float) -> Path:
    match = re.match(r"^(?P<prefix>.+?)_(?P<temp>\d+(?:\.\d+)?)(?P<suffix>(?:_.+)?)$", base_root.name)
    if match is not None:
        prefix = match.group("prefix")
        suffix = match.group("suffix")
        return base_root.with_name(f"{prefix}_{_format_temperature_suffix(temperature)}{suffix}")
    if np.isclose(float(temperature), float(deterministic_temp)):
        return base_root
    return base_root.with_name(f"{base_root.name}_{_format_temperature_suffix(temperature)}")


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
    prompt_seq = _select_prompt_sequence(prompt_hl, layer_choice=layer_choice, prompt_tail=prompt_tail)

    hk = np.asarray(npz["hk"])
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
    for file in files:
        pair_number_from_name = _pair_number_from_filename(file)
        if max_pairs is not None and pair_number_from_name is not None and pair_number_from_name > max_pairs:
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
        existing_rollout = existing.rollout_id if existing.rollout_id is not None else -1
        record_rollout = record.rollout_id if record.rollout_id is not None else -1
        if record_rollout < existing_rollout:
            chosen[record.prompt_id] = record
    return list(sorted(chosen.values(), key=lambda rec: rec.prompt_id))


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
) -> PCAProjector:
    all_activations = np.concatenate([record.activation_sequence for record in records], axis=0)
    sample_size = min(pca_sample_size, all_activations.shape[0])
    if sample_size < 2:
        raise RuntimeError("Need at least two activations to fit PCA.")
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(all_activations.shape[0], size=sample_size, replace=False)
    sampled = all_activations[sample_idx].astype(np.float64, copy=False)
    dim = min(pca_dim, sampled.shape[0], sampled.shape[1])
    mean = sampled.mean(axis=0, keepdims=True)
    centered = sampled - mean
    if centered.shape[0] <= centered.shape[1]:
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1][:dim]
        eigvals = np.maximum(eigvals[order], 1e-12)
        eigvecs = eigvecs[:, order]
        components = (centered.T @ eigvecs / np.sqrt(eigvals)).T
    else:
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1][:dim]
        components = eigvecs[:, order].T
    components = components.astype(np.float32, copy=False)
    return PCAProjector(mean_=mean.astype(np.float32, copy=False), components_=components)


def _transform_sequence(
    record: SequenceRecord,
    basis_name: str,
    pca_model: PCAProjector | None,
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


def _build_prompt_transition_matrices(
    records: Sequence[SequenceRecord],
    basis_name: str,
    hankel_window: int,
    pca_model: PCAProjector | None,
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
        hankel = _build_hankel_states(sequence, window=hankel_window)
        x_rows.append(hankel[:-1])
        y_rows.append(hankel[1:])
    x = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)
    return x, y, feature_dim


def _resolve_artifact_path(path_str: str, results_path: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    script_root = Path(__file__).resolve().parent
    candidates = [
        results_path.parent / path,
        script_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (script_root / path).resolve()


def _discover_result_specs(
    outputs_root: Path,
    results_glob: str,
    allowed_temperatures: Sequence[float],
    allowed_pair_caps: Sequence[int],
    allowed_alpha_modes: Sequence[str],
) -> List[ComboSpec]:
    specs: List[ComboSpec] = []
    skipped_filters = 0
    allowed_pair_caps_set = {int(value) for value in allowed_pair_caps}
    allowed_alpha_modes_set = {str(value) for value in allowed_alpha_modes}
    for results_path in sorted(outputs_root.glob(results_glob)):
        if not results_path.is_file():
            continue
        try:
            relative_parts = results_path.relative_to(outputs_root).parts
        except Exception:
            relative_parts = ()
        if relative_parts and relative_parts[0] in IGNORED_OUTPUT_DIR_NAMES:
            log.info("Skipping ignored output tree %s", results_path)
            continue
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Skipping unreadable JSON %s: %s", results_path, exc)
            continue
        if isinstance(payload, dict):
            results = payload.get("results", [])
        elif isinstance(payload, list):
            results = payload
        else:
            log.warning(
                "Skipping malformed results payload %s: expected object or list, got %s",
                results_path,
                type(payload).__name__,
            )
            continue
        if not isinstance(results, list):
            log.warning("Skipping malformed results payload: %s", results_path)
            continue
        pair_cap = _parse_pair_cap(results_path.parent)
        if allowed_pair_caps and pair_cap not in allowed_pair_caps_set:
            skipped_filters += 1
            log.debug("Skipping %s due to pair cap %s", results_path, pair_cap)
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            layer_choice = str(item.get("layer", "")).strip().lower()
            if layer_choice in IGNORED_LAYER_CHOICES:
                skipped_filters += 1
                continue
            if allowed_temperatures and not _matches_allowed_temperature(item.get("temperature"), allowed_temperatures):
                skipped_filters += 1
                continue
            alpha_mode = _infer_alpha_mode(item, results_path.parent)
            if allowed_alpha_modes and alpha_mode not in allowed_alpha_modes_set:
                skipped_filters += 1
                continue
            specs.append(
                ComboSpec(
                    output_dir=results_path.parent,
                    results_path=results_path,
                    payload=item,
                )
            )
    log.info(
        "Discovered %d result combos from glob %s under %s after filtering (skipped %d)",
        len(specs),
        results_glob,
        outputs_root,
        skipped_filters,
    )
    return specs


def _build_hankel_states(sequence: np.ndarray, window: int) -> np.ndarray:
    if sequence.shape[0] < window + 1:
        raise ValueError(
            f"Need at least window+1={window + 1} steps to form rollout states; got {sequence.shape[0]}."
        )
    return np.stack(
        [sequence[idx : idx + window].reshape(-1) for idx in range(sequence.shape[0] - window + 1)],
        axis=0,
    ).astype(np.float32, copy=False)


def _summarize_rollout_errors(
    sse_by_horizon: np.ndarray,
    target_sse_by_horizon: np.ndarray,
    state_counts: np.ndarray,
    scalar_counts: np.ndarray,
) -> List[RolloutMetric]:
    metrics: List[RolloutMetric] = []
    for idx in range(len(sse_by_horizon)):
        scalar_count = int(scalar_counts[idx])
        if scalar_count <= 0:
            continue
        mse = float(sse_by_horizon[idx] / scalar_count)
        relative_mse = (
            float(sse_by_horizon[idx] / target_sse_by_horizon[idx])
            if target_sse_by_horizon[idx] > 0
            else float("nan")
        )
        metrics.append(
            RolloutMetric(
                horizon=idx + 1,
                mse=mse,
                rmse=float(np.sqrt(max(mse, 0.0))),
                relative_mse=relative_mse,
                state_count=int(state_counts[idx]),
                scalar_count=scalar_count,
            )
        )
    return metrics


def _evaluate_direct_rollout(
    operator: np.ndarray,
    hankel_states: Sequence[np.ndarray],
    max_horizon: int,
) -> List[RolloutMetric]:
    if max_horizon < 1:
        return []
    operator64 = operator.astype(np.float64, copy=False)
    sse_by_horizon = np.zeros(max_horizon, dtype=np.float64)
    target_sse_by_horizon = np.zeros(max_horizon, dtype=np.float64)
    state_counts = np.zeros(max_horizon, dtype=np.int64)
    scalar_counts = np.zeros(max_horizon, dtype=np.int64)

    for states in hankel_states:
        if states.shape[0] < 2:
            continue
        prompt_horizon = min(max_horizon, states.shape[0] - 1)
        starts = states[:-1].astype(np.float64, copy=False)
        pred = starts.copy()
        for horizon in range(1, prompt_horizon + 1):
            pred = pred @ operator64
            valid = states.shape[0] - horizon
            target = states[horizon:].astype(np.float64, copy=False)
            diff = pred[:valid] - target
            idx = horizon - 1
            sse_by_horizon[idx] += float(np.sum(diff * diff))
            target_sse_by_horizon[idx] += float(np.sum(target * target))
            state_counts[idx] += int(valid)
            scalar_counts[idx] += int(target.size)
    return _summarize_rollout_errors(
        sse_by_horizon=sse_by_horizon,
        target_sse_by_horizon=target_sse_by_horizon,
        state_counts=state_counts,
        scalar_counts=scalar_counts,
    )


def _evaluate_coordinate_rollout(
    x_train: np.ndarray,
    dual_weights: np.ndarray,
    reduced_operator: np.ndarray,
    hankel_states: Sequence[np.ndarray],
    max_horizon: int,
    eval_coord_idx: np.ndarray | None,
) -> List[RolloutMetric]:
    if max_horizon < 1:
        return []
    x_train64 = x_train.astype(np.float64, copy=False)
    dual_weights64 = dual_weights.astype(np.float64, copy=False)
    reduced_operator64 = reduced_operator.astype(np.float64, copy=False)
    x_train_t = x_train64.T

    sse_by_horizon = np.zeros(max_horizon, dtype=np.float64)
    target_sse_by_horizon = np.zeros(max_horizon, dtype=np.float64)
    state_counts = np.zeros(max_horizon, dtype=np.int64)
    scalar_counts = np.zeros(max_horizon, dtype=np.int64)

    for states in hankel_states:
        if states.shape[0] < 2:
            continue
        prompt_horizon = min(max_horizon, states.shape[0] - 1)
        starts = states[:-1].astype(np.float64, copy=False)
        coeff = starts @ x_train_t
        for horizon in range(1, prompt_horizon + 1):
            pred = coeff @ dual_weights64
            valid = states.shape[0] - horizon
            target = states[horizon:].astype(np.float64, copy=False)
            if eval_coord_idx is not None:
                target = target[:, eval_coord_idx]
            diff = pred[:valid] - target
            idx = horizon - 1
            sse_by_horizon[idx] += float(np.sum(diff * diff))
            target_sse_by_horizon[idx] += float(np.sum(target * target))
            state_counts[idx] += int(valid)
            scalar_counts[idx] += int(target.size)
            coeff = coeff @ reduced_operator64
    return _summarize_rollout_errors(
        sse_by_horizon=sse_by_horizon,
        target_sse_by_horizon=target_sse_by_horizon,
        state_counts=state_counts,
        scalar_counts=scalar_counts,
    )


def _build_test_hankel_states(
    records: Sequence[SequenceRecord],
    basis_name: str,
    hankel_window: int,
    pca_model,
    fourier_low_freq_count: int,
) -> List[np.ndarray]:
    states: List[np.ndarray] = []
    for record in records:
        sequence = _transform_sequence(
            record,
            basis_name=basis_name,
            pca_model=pca_model,
            fourier_low_freq_count=fourier_low_freq_count,
        )
        states.append(_build_hankel_states(sequence, hankel_window))
    return states


def _prepare_coordinate_dual_weights(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_train64 = x_train.astype(np.float64, copy=False)
    y_train64 = y_train.astype(np.float64, copy=False)
    gram_train = x_train64 @ x_train64.T
    solve_matrix = gram_train + float(alpha) * np.eye(gram_train.shape[0], dtype=np.float64)
    return np.linalg.solve(solve_matrix, y_train64)


def _load_records_cached(
    cache: Dict[Tuple[float, str, int | None], List[SequenceRecord]],
    data_root: Path,
    temperature: float,
    layer_choice: str,
    prompt_tail: int,
    generation_steps: int,
    max_pairs: int | None,
    deterministic_temp: float,
) -> List[SequenceRecord]:
    key = (float(temperature), layer_choice, max_pairs)
    cached = cache.get(key)
    if cached is not None:
        log.debug(
            "Using cached records for tau=%.3f layer=%s max_pairs=%s (%d records)",
            float(temperature),
            layer_choice,
            "all" if max_pairs is None else max_pairs,
            len(cached),
        )
        return cached
    data_dir = _resolve_temperature_dir(
        base_root=data_root,
        deterministic_temp=deterministic_temp,
        temperature=float(temperature),
    )
    log.info(
        "Loading records from %s for tau=%.3f layer=%s max_pairs=%s",
        data_dir,
        float(temperature),
        layer_choice,
        "all" if max_pairs is None else max_pairs,
    )
    records = _load_temperature_records(
        data_dir=data_dir,
        temperature=float(temperature),
        layer_choice=layer_choice,
        prompt_tail=prompt_tail,
        generation_steps=generation_steps,
        max_pairs=max_pairs,
    )
    log.info(
        "Loaded %d records from %s for tau=%.3f layer=%s",
        len(records),
        data_dir,
        float(temperature),
        layer_choice,
    )
    cache[key] = records
    return records


def _compute_combo_metrics(
    spec: ComboSpec,
    data_root: Path,
    max_horizon: int,
    coordinate_eval_dims: int,
    records_cache: Dict[Tuple[float, str, int | None], List[SequenceRecord]],
    cfg: ExperimentConfig,
) -> Dict[str, object]:
    payload = spec.payload
    temperature = float(payload["temperature"])
    label = int(payload["label"])
    label_name = str(payload["label_name"])
    layer_choice = str(payload["layer"])
    basis_name = str(payload["basis"])
    best_alpha = float(payload["best_alpha"])
    operator_path = _resolve_artifact_path(str(payload["artifacts"]["operator_npz"]), spec.results_path)
    max_pairs = _parse_pair_cap(spec.output_dir)
    hankel_window = int(round(float(payload["operator_dim"]) / float(payload["feature_dim"])))
    fourier_low_freq_count = int(payload.get("fourier_low_freq_count", max(1, int(payload["feature_dim"]) // 2)))
    log.info(
        "Recomputing multi-step metrics for tau=%.3f label=%s layer=%s basis=%s alpha=%s",
        temperature,
        label_name,
        layer_choice,
        basis_name,
        best_alpha,
    )

    records = _load_records_cached(
        cache=records_cache,
        data_root=data_root,
        temperature=temperature,
        layer_choice=layer_choice,
        prompt_tail=5,
        generation_steps=32,
        max_pairs=max_pairs,
        deterministic_temp=cfg.deterministic_temperature,
    )
    label_records = [record for record in records if record.label == label]
    train_records, test_records = _split_train_test_promptwise(
        label_records,
        train_fraction=cfg.probe.train_fraction,
        seed=cfg.probe.random_seed,
    )
    log.debug(
        "Split %d label-specific records into %d train / %d test for tau=%.3f label=%s basis=%s",
        len(label_records),
        len(train_records),
        len(test_records),
        temperature,
        label_name,
        basis_name,
    )
    if not train_records or not test_records:
        raise RuntimeError(
            f"Need non-empty train/test splits for {spec.results_path} :: tau={temperature}, label={label_name}"
        )

    pca_model = None
    if basis_name == "pca":
        log.info(
            "Fitting PCA projector for tau=%.3f label=%s with dim=%s sample_size=%s",
            temperature,
            label_name,
            int(payload.get("pca_components_used", payload["feature_dim"])),
            int(payload.get("pca_sample_size", 10_000)),
        )
        pca_model = _fit_pca_from_training_records(
            train_records,
            pca_dim=int(payload.get("pca_components_used", payload["feature_dim"])),
            pca_sample_size=int(payload.get("pca_sample_size", 10_000)),
            seed=cfg.probe.random_seed,
        )

    test_states = _build_test_hankel_states(
        test_records,
        basis_name=basis_name,
        hankel_window=hankel_window,
        pca_model=pca_model,
        fourier_low_freq_count=fourier_low_freq_count,
    )
    total_test_states = sum(int(states.shape[0]) for states in test_states)
    log.debug(
        "Built %d test Hankel trajectories (%d total states) for tau=%.3f label=%s basis=%s",
        len(test_states),
        total_test_states,
        temperature,
        label_name,
        basis_name,
    )

    with np.load(operator_path, allow_pickle=True) as npz:
        eigvals = np.asarray(npz["eigvals"])
        if basis_name == "coordinate":
            reduced_operator = np.asarray(npz["reduced_operator"])
        else:
            koopman_matrix = np.asarray(npz["koopman_matrix"])

    if basis_name == "coordinate":
        x_train, y_train, _ = _build_prompt_transition_matrices(
            train_records,
            basis_name=basis_name,
            hankel_window=hankel_window,
            pca_model=None,
            fourier_low_freq_count=fourier_low_freq_count,
        )
        eval_coord_idx = None
        if coordinate_eval_dims > 0 and x_train.shape[1] > coordinate_eval_dims:
            rng = np.random.default_rng(cfg.probe.random_seed)
            eval_coord_idx = np.sort(rng.choice(x_train.shape[1], size=coordinate_eval_dims, replace=False))
            log.info(
                "Coordinate basis evaluation uses %d/%d Hankel coordinates",
                len(eval_coord_idx),
                x_train.shape[1],
            )
        else:
            log.info("Coordinate basis evaluation uses all %d Hankel coordinates", x_train.shape[1])
        dual_weights = _prepare_coordinate_dual_weights(x_train=x_train, y_train=y_train, alpha=best_alpha)
        metrics = _evaluate_coordinate_rollout(
            x_train=x_train,
            dual_weights=dual_weights[:, eval_coord_idx] if eval_coord_idx is not None else dual_weights,
            reduced_operator=reduced_operator,
            hankel_states=test_states,
            max_horizon=max_horizon,
            eval_coord_idx=eval_coord_idx,
        )
    else:
        metrics = _evaluate_direct_rollout(
            operator=koopman_matrix,
            hankel_states=test_states,
            max_horizon=max_horizon,
        )
    log.info(
        "Finished tau=%.3f label=%s basis=%s with %d rollout horizons",
        temperature,
        label_name,
        basis_name,
        len(metrics),
    )

    metrics_payload = [
        {
            "horizon": item.horizon,
            "mse": item.mse,
            "rmse": item.rmse,
            "relative_mse": item.relative_mse,
            "state_count": item.state_count,
            "scalar_count": item.scalar_count,
        }
        for item in metrics
    ]
    horizon_one_mse = metrics_payload[0]["mse"] if metrics_payload else None
    alpha_mode = _infer_alpha_mode(payload, spec.output_dir)
    return {
        "temperature": temperature,
        "label": label,
        "label_name": label_name,
        "layer": layer_choice,
        "basis": basis_name,
        "best_alpha": best_alpha,
        "alpha_mode": alpha_mode,
        "fit_method": payload.get("fit_method", "unknown"),
        "operator_path": str(operator_path),
        "spectral_radius": float(np.max(np.abs(eigvals))) if eigvals.size else float("nan"),
        "one_step_test_mse_from_run": float(payload["test_mse"]),
        "one_step_test_mse_recomputed": horizon_one_mse,
        "num_test_prompts": int(len(test_records)),
        "hankel_window": hankel_window,
        "coordinate_eval_dims": (
            int(len(eval_coord_idx)) if basis_name == "coordinate" and eval_coord_idx is not None else None
        ),
        "rollout_metrics": metrics_payload,
    }


def _plot_directory_summary(results: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not results or plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    raw_ax, rel_ax = axes
    for item in results:
        metrics = item.get("rollout_metrics", [])
        if not metrics:
            continue
        horizons = [entry["horizon"] for entry in metrics]
        mse = [entry["mse"] for entry in metrics]
        rel = [entry["relative_mse"] for entry in metrics]
        label = f"{item['label_name']} / {item['basis']}"
        raw_ax.plot(horizons, mse, marker="o", linewidth=1.6, markersize=3.5, label=label)
        rel_ax.plot(horizons, rel, marker="o", linewidth=1.6, markersize=3.5, label=label)

    raw_ax.set_title("Multi-step rollout MSE")
    raw_ax.set_xlabel("Horizon")
    raw_ax.set_ylabel("MSE")
    raw_ax.set_yscale("log")
    raw_ax.grid(True, alpha=0.3)

    rel_ax.set_title("Multi-step relative MSE")
    rel_ax.set_xlabel("Horizon")
    rel_ax.set_ylabel("MSE / mean(target^2)")
    rel_ax.set_yscale("log")
    rel_ax.grid(True, alpha=0.3)
    rel_ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _group_results_by_basis(results: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for item in results:
        basis_name = str(item.get("basis", "unknown"))
        grouped.setdefault(basis_name, []).append(item)
    return grouped


def run_analysis(
    outputs_root: str,
    data_root: str,
    results_glob: str,
    max_horizon: int,
    coordinate_eval_dims: int,
    allowed_temperatures: Sequence[float],
    allowed_pair_caps: Sequence[int],
    allowed_alpha_modes: Sequence[str],
) -> None:
    outputs_root_path = Path(outputs_root)
    data_root_path = Path(data_root)
    cfg = ExperimentConfig()
    specs = _discover_result_specs(
        outputs_root=outputs_root_path,
        results_glob=results_glob,
        allowed_temperatures=allowed_temperatures,
        allowed_pair_caps=allowed_pair_caps,
        allowed_alpha_modes=allowed_alpha_modes,
    )
    if not specs:
        raise FileNotFoundError(f"No experiment_results.json files found under {outputs_root_path}")

    log.info("Found %d result entries across %s", len(specs), outputs_root_path.resolve())
    records_cache: Dict[Tuple[float, str, int | None], List[SequenceRecord]] = {}
    grouped: Dict[Path, List[ComboSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.output_dir, []).append(spec)

    index_payload = {
        "outputs_root": str(outputs_root_path.resolve()),
        "directories": [],
        "basis_directories": {basis_name: [] for basis_name in VALID_BASES},
    }
    for output_dir, dir_specs in sorted(grouped.items(), key=lambda item: str(item[0])):
        log.info("Analyzing %s (%d combos)", output_dir, len(dir_specs))
        dir_results: List[Dict[str, object]] = []
        for spec in dir_specs:
            try:
                combo_metrics = _compute_combo_metrics(
                    spec=spec,
                    data_root=data_root_path,
                    max_horizon=max_horizon,
                    coordinate_eval_dims=coordinate_eval_dims,
                    records_cache=records_cache,
                    cfg=cfg,
                )
            except Exception as exc:
                log.warning("Skipping combo in %s due to error: %s", spec.results_path, exc)
                continue
            dir_results.append(combo_metrics)
            recomputed = combo_metrics["one_step_test_mse_recomputed"]
            log.info(
                "  tau=%.1f  label=%-11s  basis=%-10s  horizon1=%.4e  spectral_radius=%.4f",
                combo_metrics["temperature"],
                combo_metrics["label_name"],
                combo_metrics["basis"],
                float(recomputed) if recomputed is not None else float("nan"),
                float(combo_metrics["spectral_radius"]),
            )

        summary_path = output_dir / "multistep_rollout_results.json"
        plot_path = output_dir / "multistep_rollout_summary.png"
        _save_json(summary_path, {"results": dir_results})
        _plot_directory_summary(dir_results, plot_path)
        log.info("Saved directory multistep summary to %s", summary_path)
        basis_outputs: List[Dict[str, object]] = []
        for basis_name, basis_results in sorted(_group_results_by_basis(dir_results).items()):
            basis_summary_path = output_dir / f"multistep_rollout_results_{basis_name}.json"
            basis_plot_path = output_dir / f"multistep_rollout_summary_{basis_name}.png"
            _save_json(
                basis_summary_path,
                {
                    "basis": basis_name,
                    "results": basis_results,
                },
            )
            _plot_directory_summary(basis_results, basis_plot_path)
            log.info(
                "Saved basis-specific multistep summary for %s (%d combos) to %s",
                basis_name,
                len(basis_results),
                basis_summary_path,
            )
            basis_entry = {
                "basis": basis_name,
                "output_dir": str(output_dir.resolve()),
                "multistep_results_json": str(basis_summary_path.resolve()),
                "multistep_summary_plot": str(basis_plot_path.resolve()) if plt is not None else None,
                "combo_count": int(len(basis_results)),
            }
            basis_outputs.append(basis_entry)
            if basis_name in index_payload["basis_directories"]:
                index_payload["basis_directories"][basis_name].append(basis_entry)

        index_payload["directories"].append(
            {
                "output_dir": str(output_dir.resolve()),
                "multistep_results_json": str(summary_path.resolve()),
                "multistep_summary_plot": str(plot_path.resolve()) if plt is not None else None,
                "combo_count": int(len(dir_results)),
                "basis_outputs": basis_outputs,
            }
        )

    index_path = outputs_root_path / "multistep_rollout_index.json"
    _save_json(index_path, index_payload)
    log.info("Saved index to %s", index_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan existing run_experiment outputs and compute held-out multi-step rollout metrics "
            "for each saved Koopman operator."
        )
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory containing run_experiment output subdirectories.",
    )
    parser.add_argument(
        "--data-root",
        default="data_0_simple",
        help=(
            "Deterministic temperature data directory used to rebuild the held-out sequences. "
            "Sibling temperature directories are inferred from this path."
        ),
    )
    parser.add_argument(
        "--results-glob",
        default="**/experiment_results.json",
        help="Glob, relative to --outputs-root, used to select result JSON files.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=8,
        help="Maximum rollout horizon to score.",
    )
    parser.add_argument(
        "--coordinate-eval-dims",
        type=int,
        default=1024,
        help=(
            "For coordinate-basis outputs, evaluate rollout MSE on a reproducible random subset of "
            "Hankel coordinates. Set <= 0 to use all coordinates."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to mirror logs to a file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.7],
        help="Only analyze results for these temperatures.",
    )
    parser.add_argument(
        "--pair-caps",
        nargs="+",
        type=int,
        default=[100, 150, 200],
        help="Only analyze output directories whose parsed pair cap is in this list.",
    )
    parser.add_argument(
        "--alpha-modes",
        nargs="+",
        default=["grid_search"],
        help="Only analyze results whose alpha_mode is in this list. Default keeps non-fixed runs only.",
    )
    args = parser.parse_args()

    _setup_logging(log_file=args.log_file, log_level=args.log_level)
    run_analysis(
        outputs_root=args.outputs_root,
        data_root=args.data_root,
        results_glob=args.results_glob,
        max_horizon=args.max_horizon,
        coordinate_eval_dims=args.coordinate_eval_dims,
        allowed_temperatures=args.temperatures,
        allowed_pair_caps=args.pair_caps,
        allowed_alpha_modes=args.alpha_modes,
    )


if __name__ == "__main__":
    main()
