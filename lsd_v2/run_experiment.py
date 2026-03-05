from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from experiment.basis import AutoencoderBasis, FourierBasis, PCABasis, TopKLogitsBasis
from experiment.config import ExperimentConfig
from experiment.data import TrajectoryRecord, group_by_prompt, load_records, split_by_regime
from experiment.edmd import fit_edmd, make_hankel, one_step_pairs, spectral_features
from experiment.probe import run_linear_probe


def _flatten_hk(record: TrajectoryRecord) -> np.ndarray:
    blocks = record.hk
    return blocks.reshape(blocks.shape[0], -1)


def _derive_hl_from_hk(record: TrajectoryRecord) -> np.ndarray:
    hk = record.hk  # [K, L, T0, d]
    by_layer = np.transpose(hk, (1, 0, 2, 3))  # [L, K, T0, d]
    l, k, t0, d = by_layer.shape
    return by_layer.reshape(l, k * t0, d)


def _flatten_hl(record: TrajectoryRecord, start_layer: int) -> np.ndarray:
    hl = record.hl if record.hl is not None else _derive_hl_from_hk(record)
    if hl.shape[0] <= start_layer:
        raise ValueError(f"hl has {hl.shape[0]} layers but start_layer={start_layer}.")
    return hl[start_layer:].reshape(hl.shape[0] - start_layer, -1)


def _build_transitions_for_record(
    record: TrajectoryRecord, formulation: str, delay: int, start_layer: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if formulation == "hk":
        states = _flatten_hk(record)
        hankel = make_hankel(states, delay=delay)
        x, y = one_step_pairs(hankel)
        if record.logits is None:
            return x, y, None, None
        if record.logits.shape[0] != states.shape[0]:
            raise ValueError("logits length must match hk block length.")
        x_logits = record.logits[delay:-1]
        y_logits = record.logits[delay + 1 :]
        return x, y, x_logits, y_logits

    if formulation == "hl":
        states = _flatten_hl(record, start_layer=start_layer)
        hankel = make_hankel(states, delay=delay)
        x, y = one_step_pairs(hankel)
        return x, y, None, None

    raise ValueError(f"Unknown formulation: {formulation}")


def _stack_prompt_rollouts(
    prompt_records: Iterable[TrajectoryRecord],
    formulation: str,
    delay: int,
    start_layer: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    x_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    x_logits_all: List[np.ndarray] = []
    y_logits_all: List[np.ndarray] = []
    has_logits = False

    for record in prompt_records:
        x, y, x_logits, y_logits = _build_transitions_for_record(
            record, formulation=formulation, delay=delay, start_layer=start_layer
        )
        x_all.append(x)
        y_all.append(y)
        if x_logits is not None and y_logits is not None:
            has_logits = True
            x_logits_all.append(x_logits)
            y_logits_all.append(y_logits)

    x_cat = np.concatenate(x_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)
    if has_logits:
        return x_cat, y_cat, np.concatenate(x_logits_all, axis=0), np.concatenate(y_logits_all, axis=0)
    return x_cat, y_cat, None, None


def _basis_grid(formulation: str, cfg: ExperimentConfig):
    grid = []
    if formulation == "hk":
        grid.append(("topk_logits_256", TopKLogitsBasis(k=cfg.topk_logits_k)))
    for n_freq in cfg.fourier_low_freq_choices:
        grid.append((f"fourier_{n_freq}", FourierBasis(low_freq_count=n_freq)))
    grid.append((f"pca_{cfg.pca_dim}", PCABasis(dim=cfg.pca_dim, seed=cfg.probe.random_seed)))
    grid.append((f"ae_{cfg.autoencoder.latent_dim}", AutoencoderBasis(cfg=cfg.autoencoder)))
    return grid


def _normalize_prompt_groups(
    prompt_groups: Dict[str, List[TrajectoryRecord]],
    regime: str,
    cfg: ExperimentConfig,
) -> Dict[str, List[TrajectoryRecord]]:
    normalized: Dict[str, List[TrajectoryRecord]] = {}
    for prompt_id, recs in prompt_groups.items():
        if regime == "deterministic":
            normalized[prompt_id] = [recs[0]]
            continue
        if regime == "stochastic":
            sorted_recs = sorted(
                recs,
                key=lambda r: (r.rollout_id if r.rollout_id is not None else 10**9),
            )
            if len(sorted_recs) < cfg.stochastic_rollouts:
                continue
            normalized[prompt_id] = sorted_recs[: cfg.stochastic_rollouts]
            continue
        raise ValueError(f"Unknown regime: {regime}")
    return normalized


def _run_combo(
    prompt_groups: Dict[str, List[TrajectoryRecord]],
    formulation: str,
    regime: str,
    delay: int,
    basis_name: str,
    basis,
    cfg: ExperimentConfig,
) -> Dict[str, object]:
    spec_features: List[np.ndarray] = []
    labels: List[int] = []
    train_mse: List[float] = []
    test_mse: List[float] = []

    for prompt_id, prompt_records in prompt_groups.items():
        label = prompt_records[0].label
        x, y, x_logits, y_logits = _stack_prompt_rollouts(
            prompt_records,
            formulation=formulation,
            delay=delay,
            start_layer=cfg.hl_start_layer,
        )
        if basis_name.startswith("topk") and (x_logits is None or y_logits is None):
            continue

        result = fit_edmd(
            x=x,
            y=y,
            basis=basis,
            x_logits=x_logits,
            y_logits=y_logits,
        )
        spec_features.append(spectral_features(result.eigvals, top_n=32))
        labels.append(label)
        train_mse.append(result.train_mse)
        if result.test_mse is not None:
            test_mse.append(result.test_mse)

    if len(spec_features) < 2:
        raise RuntimeError(
            f"Not enough prompt-level features for combo {formulation}/{regime}/{delay}/{basis_name}."
        )

    x_probe = np.vstack(spec_features)
    y_probe = np.asarray(labels)
    probe = run_linear_probe(x_probe, y_probe, cfg=cfg.probe)

    return {
        "formulation": formulation,
        "regime": regime,
        "delay": delay,
        "basis": basis_name,
        "num_prompts_used": int(len(spec_features)),
        "train_feature_mse_mean": float(np.mean(train_mse)),
        "test_feature_mse_mean": float(np.mean(test_mse)) if test_mse else None,
        "probe_best_c": probe.best_c,
        "probe_test_accuracy": probe.test_accuracy,
        "probe_test_f1": probe.test_f1,
    }


def run_full_experiment(data_root: str, output_dir: str) -> None:
    cfg = ExperimentConfig()
    cfg.validate()

    records = load_records(data_root)
    regime_split = split_by_regime(
        records,
        deterministic_temp=cfg.deterministic_temperature,
        stochastic_temp=cfg.stochastic_temperature,
    )
    if len(regime_split["deterministic"]) == 0:
        raise RuntimeError("No deterministic records found at temperature=0.0.")
    if len(regime_split["stochastic"]) == 0:
        raise RuntimeError("No stochastic records found at temperature=0.7.")

    results: List[Dict[str, object]] = []
    for formulation in ("hk", "hl"):
        delays = cfg.hk_delays if formulation == "hk" else cfg.hl_delays
        for regime, records_in_regime in regime_split.items():
            prompt_groups = _normalize_prompt_groups(
                group_by_prompt(records_in_regime), regime=regime, cfg=cfg
            )
            for delay in delays:
                for basis_name, basis in _basis_grid(formulation=formulation, cfg=cfg):
                    combo_result = _run_combo(
                        prompt_groups=prompt_groups,
                        formulation=formulation,
                        regime=regime,
                        delay=delay,
                        basis_name=basis_name,
                        basis=basis,
                        cfg=cfg,
                    )
                    results.append(combo_result)
                    print(
                        f"[done] formulation={formulation} regime={regime} "
                        f"delay={delay} basis={basis_name} acc={combo_result['probe_test_accuracy']:.4f}"
                    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STAT 31310 Koopman experiment.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Directory containing .npz trajectory files (default: data).",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory to write results JSON.")
    args = parser.parse_args()
    run_full_experiment(data_root=args.data_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

