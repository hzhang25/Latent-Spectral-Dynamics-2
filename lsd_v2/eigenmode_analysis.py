from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Keep BLAS thread counts small for eigendecomposition-heavy analysis on shared nodes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/eagle/GeoSolv/qinan/LSD/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

from experiment.config import ExperimentConfig

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
except Exception:  # pragma: no cover - runtime dependency
    snapshot_download = None
    EntryNotFoundError = Exception
    LocalEntryNotFoundError = Exception

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - runtime dependency
    safe_open = None

try:
    import torch
except Exception:  # pragma: no cover - runtime dependency
    torch = None

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - runtime dependency
    AutoTokenizer = None


log = logging.getLogger("lsd.eigenmode")
VALID_BASES = ["coordinate", "pca", "fourier"]
IGNORED_OUTPUT_DIR_NAMES = {"tier1", "last_token_pca"}
IGNORED_LAYER_CHOICES = {"all"}


@dataclass(frozen=True)
class ComboSpec:
    output_dir: Path
    results_path: Path
    payload: Dict[str, object]


@dataclass(frozen=True)
class ModeSummary:
    dominant_index: int
    dominant_eigenvalue: complex
    vector: np.ndarray


def _setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("lsd.eigenmode")
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


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _complex_to_dict(value: complex) -> Dict[str, float]:
    return {
        "real": float(np.real(value)),
        "imag": float(np.imag(value)),
        "magnitude": float(np.abs(value)),
        "angle": float(np.angle(value)),
    }


def _parse_pair_cap(output_dir: Path) -> int | None:
    tail = output_dir.name.rsplit("_", 1)[-1]
    if tail == "all":
        return None
    if tail.isdigit():
        return int(tail)
    return None


def _matches_allowed_temperature(value: object, allowed_temperatures: Sequence[float]) -> bool:
    try:
        temperature = float(value)
    except Exception:
        return False
    return any(np.isclose(temperature, float(candidate)) for candidate in allowed_temperatures)


def _infer_alpha_mode(item: Dict[str, object], output_dir: Path) -> str:
    raw_mode = item.get("alpha_mode")
    if raw_mode is not None:
        text = str(raw_mode).strip()
        if text:
            return text
    if any(part.startswith("alpha") for part in output_dir.name.split("_")):
        return "fixed"
    return "grid_search"


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
    allowed_pair_caps_set = {int(value) for value in allowed_pair_caps}
    allowed_alpha_modes_set = {str(value) for value in allowed_alpha_modes}
    skipped_filters = 0
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
        "Filtered eigenmode analysis to temperatures=%s pair_caps=%s alpha_modes=%s",
        [float(value) for value in allowed_temperatures],
        [int(value) for value in allowed_pair_caps],
        [str(value) for value in allowed_alpha_modes],
    )
    log.info(
        "Discovered %d result combos from glob %s under %s after filtering (skipped %d)",
        len(specs),
        results_glob,
        outputs_root,
        skipped_filters,
    )
    return specs


def _compute_direct_mode_summary(operator_path: Path) -> ModeSummary:
    with np.load(operator_path, allow_pickle=True) as npz:
        if "koopman_matrix" not in npz:
            raise KeyError(f"{operator_path} is missing koopman_matrix")
        operator = np.asarray(npz["koopman_matrix"], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eig(operator)
    dominant_index = int(np.argmin(np.abs(eigvals - 1.0)))
    return ModeSummary(
        dominant_index=dominant_index,
        dominant_eigenvalue=complex(eigvals[dominant_index]),
        vector=np.asarray(eigvecs[:, dominant_index], dtype=np.complex128),
    )


def _compute_coordinate_mode_summary(operator_path: Path) -> ModeSummary:
    with np.load(operator_path, allow_pickle=True) as npz:
        eigvals = np.asarray(npz["eigvals"], dtype=np.complex128)
        if "primal_modes" not in npz:
            raise KeyError(f"{operator_path} is missing primal_modes")
        primal_modes = np.asarray(npz["primal_modes"], dtype=np.complex128)
    if primal_modes.ndim != 2 or primal_modes.shape[1] == 0:
        raise ValueError(f"{operator_path} has invalid primal_modes shape {primal_modes.shape}")
    dominant_index = int(np.argmin(np.abs(eigvals - 1.0)))
    if dominant_index >= primal_modes.shape[1]:
        raise IndexError(
            f"{operator_path}: dominant eigen-index {dominant_index} exceeds saved mode count {primal_modes.shape[1]}"
        )
    return ModeSummary(
        dominant_index=dominant_index,
        dominant_eigenvalue=complex(eigvals[dominant_index]),
        vector=np.asarray(primal_modes[:, dominant_index], dtype=np.complex128),
    )


def _phase_invariant_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom < 1e-12:
        return float("nan")
    return float(np.abs(np.vdot(vec_a, vec_b)) / denom)


def _phase_align_to_largest_coordinate(vector: np.ndarray) -> np.ndarray:
    if vector.size == 0:
        return vector.astype(np.complex128, copy=False)
    ref_idx = int(np.argmax(np.abs(vector)))
    phase = np.angle(vector[ref_idx])
    return vector * np.exp(-1j * phase)


def _resolve_model(model_name: str) -> str:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required for vocabulary projection.")
    try:
        path = snapshot_download(model_name, local_files_only=True)
        snap = Path(path)
        if any(snap.glob("*.safetensors")) or any(snap.glob("*.bin")):
            log.info("Resolved cached model snapshot for %s at %s", model_name, path)
            return path
    except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
        pass
    log.info("Downloading model snapshot for %s via HuggingFace cache", model_name)
    return snapshot_download(model_name)


def _load_tensor_from_safetensors(model_dir: Path, candidate_keys: Sequence[str]) -> np.ndarray:
    if safe_open is None:
        raise RuntimeError("safetensors is required to load the unembedding matrix efficiently.")

    def _read_tensor(shard_path: Path, key: str) -> np.ndarray:
        if torch is not None:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(key)
                return tensor.to(dtype=torch.float32).cpu().numpy()
        with safe_open(str(shard_path), framework="np", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            return np.asarray(tensor, dtype=np.float32)

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in candidate_keys:
            shard_name = weight_map.get(key)
            if shard_name is None:
                continue
            shard_path = model_dir / shard_name
            log.info("Loading tensor %s from shard %s", key, shard_path.name)
            return _read_tensor(shard_path, key)
    for shard_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="np", device="cpu") as handle:
            keys = set(handle.keys())
            for key in candidate_keys:
                if key in keys:
                    log.info("Loading tensor %s from shard %s", key, shard_path.name)
                    return _read_tensor(shard_path, key)
    raise FileNotFoundError(
        f"Could not find any of {candidate_keys} in safetensors files under {model_dir}"
    )


def _load_tokenizer_and_unembedding(model_name: str) -> tuple[object, np.ndarray]:
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required for vocabulary projection.")
    model_path = Path(_resolve_model(model_name))
    log.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    unembed = _load_tensor_from_safetensors(
        model_path,
        candidate_keys=("lm_head.weight", "model.embed_tokens.weight"),
    )
    return tokenizer, unembed


def _project_mode_to_vocabulary(
    mode_vector: np.ndarray,
    feature_dim: int,
    hankel_window: int,
    token_index: int,
    unembed: np.ndarray,
    tokenizer,
    top_k: int,
) -> Dict[str, object]:
    if feature_dim != int(unembed.shape[1]):
        raise ValueError(
            f"Feature dim {feature_dim} does not match unembedding hidden dim {unembed.shape[1]}"
        )
    if mode_vector.size != feature_dim * hankel_window:
        raise ValueError(
            f"Mode length {mode_vector.size} does not match feature_dim*window={feature_dim * hankel_window}"
        )
    token_slot = token_index if token_index >= 0 else hankel_window + token_index
    if token_slot < 0 or token_slot >= hankel_window:
        raise IndexError(f"token_index={token_index} resolves to invalid slot {token_slot} for window {hankel_window}")
    start = token_slot * feature_dim
    stop = start + feature_dim
    token_slice = np.asarray(mode_vector[start:stop], dtype=np.complex128)
    aligned = _phase_align_to_largest_coordinate(token_slice)
    slice_real = aligned.real.astype(np.float32, copy=False)
    logits = unembed @ slice_real
    top_k = min(int(top_k), int(logits.shape[0]))
    top_idx = np.argpartition(logits, kth=-top_k)[-top_k:]
    top_idx = top_idx[np.argsort(logits[top_idx])[::-1]]
    top_tokens: List[Dict[str, object]] = []
    for idx in top_idx:
        token_id = int(idx)
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        raw_token = None
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            try:
                raw_token = tokenizer.convert_ids_to_tokens(token_id)
            except Exception:
                raw_token = None
        top_tokens.append(
            {
                "rank": len(top_tokens) + 1,
                "token_id": token_id,
                "decoded": decoded,
                "raw_token": raw_token,
                "logit_score": float(logits[token_id]),
            }
        )
    imag_norm = float(np.linalg.norm(aligned.imag))
    real_norm = float(np.linalg.norm(aligned.real))
    return {
        "hankel_token_slot": int(token_slot),
        "imag_to_real_norm_ratio": float(imag_norm / real_norm) if real_norm > 1e-12 else float("inf"),
        "top_tokens": top_tokens,
    }


def _group_specs_for_comparison(specs: Sequence[ComboSpec]) -> Dict[Tuple[Path, float, str, str], Dict[str, ComboSpec]]:
    grouped: Dict[Tuple[Path, float, str, str], Dict[str, ComboSpec]] = {}
    for spec in specs:
        payload = spec.payload
        label_name = str(payload.get("label_name", "")).strip().lower()
        if label_name not in {"truthful", "untruthful"}:
            continue
        key = (
            spec.output_dir,
            float(payload["temperature"]),
            str(payload["layer"]),
            str(payload["basis"]),
        )
        grouped.setdefault(key, {})[label_name] = spec
    log.info("Grouped %d result entries into %d comparison buckets", len(specs), len(grouped))
    return grouped


def run_analysis(
    outputs_root: str,
    results_glob: str,
    temperatures: Sequence[float],
    pair_caps: Sequence[int],
    alpha_modes: Sequence[str],
    model_name: str,
    projection_top_k: int,
    projection_token_index: int,
) -> None:
    outputs_root_path = Path(outputs_root)
    cfg = ExperimentConfig()
    log.info("Starting eigenmode analysis under %s", outputs_root_path.resolve())
    specs = _discover_result_specs(
        outputs_root=outputs_root_path,
        results_glob=results_glob,
        allowed_temperatures=temperatures,
        allowed_pair_caps=pair_caps,
        allowed_alpha_modes=alpha_modes,
    )
    if not specs:
        raise FileNotFoundError(f"No experiment_results.json files found under {outputs_root_path}")
    grouped = _group_specs_for_comparison(specs)
    log.info("Prepared %d truth-vs-lie comparison groups", len(grouped))

    need_vocab_projection = any(
        key[2] == "final" and key[3] == "coordinate" and "truthful" in item and "untruthful" in item
        for key, item in grouped.items()
    )
    tokenizer = None
    unembed = None
    if need_vocab_projection:
        log.info("Loading tokenizer and unembedding matrix for final-layer coordinate vocabulary projection")
        tokenizer, unembed = _load_tokenizer_and_unembedding(model_name=model_name or cfg.model_name)
        log.info("Unembedding matrix loaded with shape %s", tuple(unembed.shape))
    else:
        log.info("No final-layer coordinate runs matched filters; skipping vocabulary projection setup")

    index_entries: List[Dict[str, object]] = []
    by_dir: Dict[Path, List[Dict[str, object]]] = {}

    for (output_dir, temperature, layer_choice, basis_name), pair in sorted(grouped.items(), key=lambda item: str(item[0])):
        if "truthful" not in pair or "untruthful" not in pair:
            log.debug("Skipping incomplete pair for %s", (output_dir, temperature, layer_choice, basis_name))
            continue
        log.info(
            "Comparing truth vs lie for tau=%.1f layer=%s basis=%s in %s",
            temperature,
            layer_choice,
            basis_name,
            output_dir,
        )
        truth_spec = pair["truthful"]
        lie_spec = pair["untruthful"]
        truth_payload = truth_spec.payload
        lie_payload = lie_spec.payload
        truth_operator = _resolve_artifact_path(str(truth_payload["artifacts"]["operator_npz"]), truth_spec.results_path)
        lie_operator = _resolve_artifact_path(str(lie_payload["artifacts"]["operator_npz"]), lie_spec.results_path)

        if basis_name == "coordinate":
            log.debug("Loading lifted coordinate modes from %s and %s", truth_operator, lie_operator)
            truth_mode = _compute_coordinate_mode_summary(truth_operator)
            lie_mode = _compute_coordinate_mode_summary(lie_operator)
        else:
            log.debug("Loading direct Koopman matrices from %s and %s", truth_operator, lie_operator)
            truth_mode = _compute_direct_mode_summary(truth_operator)
            lie_mode = _compute_direct_mode_summary(lie_operator)

        similarity = _phase_invariant_cosine_similarity(truth_mode.vector, lie_mode.vector)
        pair_cap = _parse_pair_cap(output_dir)
        alpha_mode = _infer_alpha_mode(truth_payload, output_dir)
        result_entry: Dict[str, object] = {
            "output_dir": str(output_dir.resolve()),
            "temperature": float(temperature),
            "pair_cap": pair_cap,
            "layer": layer_choice,
            "basis": basis_name,
            "alpha_mode": alpha_mode,
            "dominant_mode_cosine_similarity": similarity,
            "truth": {
                "operator_path": str(truth_operator),
                "dominant_index": int(truth_mode.dominant_index),
                "dominant_eigenvalue": _complex_to_dict(truth_mode.dominant_eigenvalue),
            },
            "lie": {
                "operator_path": str(lie_operator),
                "dominant_index": int(lie_mode.dominant_index),
                "dominant_eigenvalue": _complex_to_dict(lie_mode.dominant_eigenvalue),
            },
        }

        if basis_name == "coordinate" and layer_choice == "final":
            if tokenizer is None or unembed is None:
                raise RuntimeError("Tokenizer/unembedding matrix was not initialized for vocabulary projection.")
            feature_dim = int(truth_payload["feature_dim"])
            hankel_window = int(round(float(truth_payload["operator_dim"]) / float(truth_payload["feature_dim"])))
            log.info(
                "Projecting final-layer coordinate dominant modes to vocabulary space (feature_dim=%d, hankel_window=%d, token_slot=%d)",
                feature_dim,
                hankel_window,
                projection_token_index,
            )
            result_entry["vocabulary_projection"] = {
                "token_index": int(projection_token_index),
                "truth": _project_mode_to_vocabulary(
                    mode_vector=truth_mode.vector,
                    feature_dim=feature_dim,
                    hankel_window=hankel_window,
                    token_index=projection_token_index,
                    unembed=unembed,
                    tokenizer=tokenizer,
                    top_k=projection_top_k,
                ),
                "lie": _project_mode_to_vocabulary(
                    mode_vector=lie_mode.vector,
                    feature_dim=feature_dim,
                    hankel_window=hankel_window,
                    token_index=projection_token_index,
                    unembed=unembed,
                    tokenizer=tokenizer,
                    top_k=projection_top_k,
                ),
            }

        by_dir.setdefault(output_dir, []).append(result_entry)
        index_entries.append(result_entry)
        log.info(
            "Analyzed tau=%.1f layer=%s basis=%s p=%s cosine=%.4f",
            temperature,
            layer_choice,
            basis_name,
            "all" if pair_cap is None else pair_cap,
            similarity,
        )

    for output_dir, entries in sorted(by_dir.items(), key=lambda item: str(item[0])):
        summary_path = output_dir / "eigenmode_analysis.json"
        _save_json(summary_path, {"results": entries})
        log.info("Saved eigenmode analysis for %s to %s", output_dir, summary_path)

    index_path = outputs_root_path / "eigenmode_analysis_index.json"
    _save_json(
        index_path,
        {
            "outputs_root": str(outputs_root_path.resolve()),
            "result_count": int(len(index_entries)),
            "results": index_entries,
        },
    )
    log.info("Saved global eigenmode analysis index with %d entries to %s", len(index_entries), index_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dominant Koopman eigenmodes between truthful and untruthful runs, "
            "and optionally project final-layer coordinate modes back into vocabulary space."
        )
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory containing run_experiment output subdirectories.",
    )
    parser.add_argument(
        "--results-glob",
        default="**/experiment_results.json",
        help="Glob, relative to --outputs-root, used to select result JSON files.",
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
        help="Only analyze results whose alpha_mode is in this list.",
    )
    parser.add_argument(
        "--model-name",
        default=ExperimentConfig().model_name,
        help="HF model name or local snapshot path used to load the tokenizer and unembedding matrix.",
    )
    parser.add_argument(
        "--projection-top-k",
        type=int,
        default=5,
        help="Number of top decoded tokens to report for final-layer coordinate dominant modes.",
    )
    parser.add_argument(
        "--projection-token-index",
        type=int,
        default=-1,
        help=(
            "Which token slot inside the Hankel window to project for vocabulary grounding. "
            "Default -1 uses the most recent token in the window."
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
    args = parser.parse_args()

    _setup_logging(log_file=args.log_file, log_level=args.log_level)
    run_analysis(
        outputs_root=args.outputs_root,
        results_glob=args.results_glob,
        temperatures=args.temperatures,
        pair_caps=args.pair_caps,
        alpha_modes=args.alpha_modes,
        model_name=args.model_name,
        projection_top_k=args.projection_top_k,
        projection_token_index=args.projection_token_index,
    )


if __name__ == "__main__":
    main()
