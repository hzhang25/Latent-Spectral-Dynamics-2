from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment.data import TrajectoryRecord, _parse_label


def _setup_logging(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("last_token_pca")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


log = logging.getLogger("last_token_pca")


def _flatten_recorded_token_activations(record: TrajectoryRecord) -> np.ndarray:
    """
    Flatten the recorded per-step token activations to shape [num_recorded_tokens, L, d].

    Important: this dataset stores one token state per forward step, not arbitrary
    prompt-token positions. Therefore:
      - token step 0 = last prompt token from the prefill pass
      - token step 1 = first generated token
      - token step 2 = second generated token
      - ...
    """
    if record.hk.ndim != 4:
        raise ValueError(f"Expected hk with 4 dims, got shape {record.hk.shape}")
    return np.transpose(record.hk, (0, 2, 1, 3)).reshape(-1, record.hk.shape[1], record.hk.shape[3])


def _resolve_token_index(token_index: int, total_tokens: int) -> int:
    resolved = token_index if token_index >= 0 else total_tokens + token_index
    if not 0 <= resolved < total_tokens:
        raise ValueError(
            f"Token index {token_index} is out of range for {total_tokens} available tokens."
        )
    return resolved


def _recorded_token_description(token_index: int) -> str:
    if token_index == 0:
        return "last_prompt_token"
    if token_index == 1:
        return "first_generated_token"
    if token_index > 1:
        return f"generated_token_{token_index}"
    return f"token_step_{token_index}"


def _input_token_description(token_index: int, total_tokens: int) -> str:
    if token_index == 0:
        return "first_input_token"
    if token_index == total_tokens - 1:
        return "last_input_token"
    return f"input_token_{token_index}"


def _extract_recorded_token_activations(record: TrajectoryRecord, token_index: int) -> np.ndarray:
    token_activations = _flatten_recorded_token_activations(record)
    resolved = _resolve_token_index(token_index, token_activations.shape[0])
    return token_activations[resolved]


def _extract_input_token_activations(record: TrajectoryRecord, token_index: int) -> np.ndarray:
    if record.prompt_hl is not None:
        resolved = _resolve_token_index(token_index, record.prompt_hl.shape[1])
        return record.prompt_hl[:, resolved, :]

    if record.input_token_ids is None:
        raise RuntimeError(
            "Input-token selection requires records with 'prompt_hl' saved in the .npz files. "
            "Regenerate the dataset with the updated extract_data.py."
        )

    resolved = _resolve_token_index(token_index, len(record.input_token_ids))
    if resolved == len(record.input_token_ids) - 1:
        return _extract_recorded_token_activations(record, 0)

    raise RuntimeError(
        "This dataset does not contain arbitrary prompt-token hidden states. "
        "Only the last input token is recoverable from older files. "
        "Regenerate the dataset with the updated extract_data.py to analyze other input tokens."
    )


def _resolve_token_selection(
    record: TrajectoryRecord,
    token_index: int,
    input_token_index: int | None,
) -> tuple[np.ndarray, int, str, str, str]:
    if input_token_index is not None:
        total_input_tokens = (
            record.prompt_hl.shape[1]
            if record.prompt_hl is not None
            else len(record.input_token_ids) if record.input_token_ids is not None else 0
        )
        resolved_input_index = _resolve_token_index(input_token_index, total_input_tokens)
        token_description = _input_token_description(resolved_input_index, total_input_tokens)
        token_label = token_description.replace("_", " ")
        return (
            _extract_input_token_activations(record, input_token_index),
            resolved_input_index,
            token_description,
            token_label,
            "input",
        )

    total_recorded_tokens = _flatten_recorded_token_activations(record).shape[0]
    resolved_token_index = _resolve_token_index(token_index, total_recorded_tokens)
    token_description = _recorded_token_description(resolved_token_index)
    token_label = token_description.replace("_", " ")
    return (
        _extract_recorded_token_activations(record, token_index),
        resolved_token_index,
        token_description,
        token_label,
        "recorded",
    )


def _load_limited_records(data_root: str, max_prompts: int, temperature: float) -> List[TrajectoryRecord]:
    """
    Stream records from disk and stop once we have `max_prompts` matching samples.
    This avoids loading the entire dataset into memory first.
    """
    root = Path(data_root)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {root}")

    log.info("Found %d total .npz files under %s.", len(files), root.resolve())
    selected: List[TrajectoryRecord] = []
    scanned = 0
    for file in files:
        scanned += 1
        with np.load(file, allow_pickle=True) as npz:
            temp = float(npz["temperature"].item())
            if not np.isclose(temp, temperature):
                continue

            selected.append(
                TrajectoryRecord(
                    prompt_id=str(npz["prompt_id"].item()),
                    label=_parse_label(npz["label"]),
                    temperature=temp,
                    rollout_id=int(npz["rollout_id"].item()) if "rollout_id" in npz else None,
                    hk=np.asarray(npz["hk"]),
                    hl=np.asarray(npz["hl"]) if "hl" in npz else None,
                    logits=np.asarray(npz["logits"]) if "logits" in npz else None,
                    prompt_hl=np.asarray(npz["prompt_hl"]) if "prompt_hl" in npz else None,
                    input_token_ids=(
                        np.asarray(npz["input_token_ids"]) if "input_token_ids" in npz else None
                    ),
                    input_tok_len_raw=(
                        int(npz["input_tok_len_raw"].item()) if "input_tok_len_raw" in npz else None
                    ),
                )
            )
        if len(selected) >= max_prompts:
            break

    log.info(
        "Scanned %d files to collect %d records at temperature=%.3f.",
        scanned,
        len(selected),
        temperature,
    )

    if not selected:
        raise RuntimeError(f"No records found at temperature={temperature}.")
    if len(selected) < 2:
        raise RuntimeError(f"Need at least 2 prompts, found only {len(selected)}.")
    labels = {record.label for record in selected}
    if labels != {0, 1}:
        raise RuntimeError(
            "Selected prompts do not contain both truthful and untruthful labels. "
            "Increase --max-prompts or reorder the files."
        )
    return selected


def _layer_metrics(last_token_by_layer: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "l2_norm": np.linalg.norm(last_token_by_layer, axis=1),
        "mean_abs": np.mean(np.abs(last_token_by_layer), axis=1),
        "signed_mean": np.mean(last_token_by_layer, axis=1),
    }


def _layer_cosine_matrix(token_by_layer: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(token_by_layer, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normalized = token_by_layer / norms
    return normalized @ normalized.T


def _orthogonality_metrics(activations: np.ndarray) -> dict[str, np.ndarray]:
    cosine_matrices = np.stack([_layer_cosine_matrix(act) for act in activations], axis=0)
    num_layers = cosine_matrices.shape[1]
    off_diag_mask = ~np.eye(num_layers, dtype=bool)
    off_diag = cosine_matrices[:, off_diag_mask]
    return {
        "cosine_matrices": cosine_matrices,
        "mean_abs_offdiag_cosine": np.mean(np.abs(off_diag), axis=1),
        "mean_offdiag_cosine": np.mean(off_diag, axis=1),
        "max_abs_offdiag_cosine": np.max(np.abs(off_diag), axis=1),
    }


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def _plot_pca_scatter(
    pca_projection: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    explained_variance_ratio: np.ndarray,
    out_path: Path,
    token_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    split_to_marker = [("train", train_indices, "o"), ("test", test_indices, "x")]
    label_to_color = [(1, "truthful", "tab:blue"), (0, "untruthful", "tab:orange")]

    for split_name, split_indices, marker in split_to_marker:
        for label, label_name, color in label_to_color:
            mask = y[split_indices] == label
            if not np.any(mask):
                continue
            coords = pca_projection[split_indices][mask]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=color,
                marker=marker,
                alpha=0.8,
                label=f"{label_name} ({split_name})",
            )

    x_var = 100.0 * float(explained_variance_ratio[0]) if len(explained_variance_ratio) >= 1 else 0.0
    y_var = 100.0 * float(explained_variance_ratio[1]) if len(explained_variance_ratio) >= 2 else 0.0
    ax.set_xlabel(f"PC1 ({x_var:.1f}% var)")
    ax.set_ylabel(f"PC2 ({y_var:.1f}% var)")
    ax.set_title(f"{token_label} PCA projection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_layerwise_pca_progression(
    layerwise_pca_projection: np.ndarray,
    y: np.ndarray,
    explained_variance_ratio: np.ndarray,
    out_path: Path,
    token_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    num_layers = layerwise_pca_projection.shape[1]
    layers = np.arange(num_layers)
    stride = max(1, num_layers // 8)

    style_by_label = {
        1: {"name": "truthful", "color": "tab:blue"},
        0: {"name": "untruthful", "color": "tab:orange"},
    }

    for label, style in style_by_label.items():
        class_paths = layerwise_pca_projection[y == label]
        if len(class_paths) == 0:
            continue

        for path in class_paths[: min(8, len(class_paths))]:
            ax.plot(path[:, 0], path[:, 1], color=style["color"], alpha=0.12, linewidth=1.0)

        mean_path = class_paths.mean(axis=0)
        ax.plot(mean_path[:, 0], mean_path[:, 1], color=style["color"], linewidth=2.5, label=style["name"])
        ax.scatter(mean_path[:, 0], mean_path[:, 1], color=style["color"], s=28, alpha=0.9)

        for layer_idx in range(0, num_layers, stride):
            ax.annotate(
                str(layer_idx),
                (mean_path[layer_idx, 0], mean_path[layer_idx, 1]),
                color=style["color"],
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

        ax.annotate(
            f"{style['name']} start",
            (mean_path[0, 0], mean_path[0, 1]),
            color=style["color"],
            fontsize=9,
            xytext=(6, 6),
            textcoords="offset points",
        )
        ax.annotate(
            f"{style['name']} end",
            (mean_path[-1, 0], mean_path[-1, 1]),
            color=style["color"],
            fontsize=9,
            xytext=(6, -10),
            textcoords="offset points",
        )

    x_var = 100.0 * float(explained_variance_ratio[0]) if len(explained_variance_ratio) >= 1 else 0.0
    y_var = 100.0 * float(explained_variance_ratio[1]) if len(explained_variance_ratio) >= 2 else 0.0
    ax.set_xlabel(f"PC1 ({x_var:.1f}% var)")
    ax.set_ylabel(f"PC2 ({y_var:.1f}% var)")
    ax.set_title(f"Layerwise {token_label} PCA progression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_pca_linear_probe(
    activations: np.ndarray,
    y: np.ndarray,
    random_seed: int,
    train_fraction: float = 0.8,
    max_pca_components: int = 8,
) -> dict[str, Any]:
    if activations.ndim != 3:
        raise ValueError(f"Expected activations with shape [N, L, D], got {activations.shape}")

    x = activations.reshape(activations.shape[0], -1)
    class_counts = np.bincount(y, minlength=2)
    if int(class_counts.min()) < 2:
        raise RuntimeError(
            "Need at least 2 samples per class for an 80/20 stratified split. "
            f"Counts: truthful={int(class_counts[1])}, untruthful={int(class_counts[0])}."
        )

    indices = np.arange(len(y))
    try:
        train_indices, test_indices = train_test_split(
            indices,
            train_size=train_fraction,
            random_state=random_seed,
            stratify=y,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Unable to create a stratified 80/20 split for truthful vs untruthful prompts. "
            "Increase --max-prompts or ensure both labels are sufficiently represented."
        ) from exc
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    x_train_std, x_test_std, mean, std = _standardize_train_test(x_train, x_test)
    max_rank = max(1, min(x_train_std.shape[0], x_train_std.shape[1]) - 1)
    n_components = max(1, min(max_pca_components, max_rank))

    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_seed)
    pca.fit(x_train_std)
    train_pca = pca.transform(x_train_std)
    test_pca = pca.transform(x_test_std)

    probe = LogisticRegression(max_iter=2000, random_state=random_seed)
    probe.fit(train_pca, y_train)
    test_pred = probe.predict(test_pca)
    test_prob = probe.predict_proba(test_pca)[:, 1]

    x_all_std = (x - mean) / std
    pca_projection = pca.transform(x_all_std)
    if pca_projection.shape[1] == 1:
        pca_projection = np.concatenate(
            [pca_projection, np.zeros((pca_projection.shape[0], 1), dtype=pca_projection.dtype)],
            axis=1,
        )

    layer_features_train = activations[train_indices].reshape(-1, activations.shape[-1])
    layer_features_all = activations.reshape(-1, activations.shape[-1])
    layer_mean = layer_features_train.mean(axis=0, keepdims=True)
    layer_std = layer_features_train.std(axis=0, keepdims=True)
    layer_std = np.where(layer_std < 1e-8, 1.0, layer_std)
    layer_features_train_std = (layer_features_train - layer_mean) / layer_std
    layer_features_all_std = (layer_features_all - layer_mean) / layer_std

    layer_max_rank = max(1, min(layer_features_train_std.shape[0], layer_features_train_std.shape[1]) - 1)
    layer_n_components = max(1, min(2, layer_max_rank))
    layer_pca = PCA(n_components=layer_n_components, svd_solver="full", random_state=random_seed)
    layer_pca.fit(layer_features_train_std)
    layerwise_pca_projection = layer_pca.transform(layer_features_all_std)
    if layerwise_pca_projection.shape[1] == 1:
        layerwise_pca_projection = np.concatenate(
            [
                layerwise_pca_projection,
                np.zeros((layerwise_pca_projection.shape[0], 1), dtype=layerwise_pca_projection.dtype),
            ],
            axis=1,
        )
    layerwise_pca_projection = layerwise_pca_projection.reshape(activations.shape[0], activations.shape[1], 2)

    return {
        "flat_activations": x,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "y_train": y_train,
        "y_test": y_test,
        "train_pca": train_pca,
        "test_pca": test_pca,
        "pca_projection": pca_projection[:, :2],
        "layerwise_pca_projection": layerwise_pca_projection,
        "probe_test_pred": test_pred,
        "probe_test_prob": test_prob,
        "summary": {
            "train_fraction": float(train_fraction),
            "test_fraction": float(1.0 - train_fraction),
            "random_seed": int(random_seed),
            "train_size": int(len(train_indices)),
            "test_size": int(len(test_indices)),
            "pca_components": int(n_components),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "explained_variance_cumulative": np.cumsum(pca.explained_variance_ratio_).astype(float).tolist(),
            "probe_accuracy": float(accuracy_score(y_test, test_pred)),
            "probe_f1": float(f1_score(y_test, test_pred)),
            "probe_confusion": confusion_matrix(y_test, test_pred, labels=[0, 1]).astype(int).tolist(),
            "probe_coefficients": probe.coef_.astype(float).tolist(),
            "probe_intercept": probe.intercept_.astype(float).tolist(),
            "layerwise_pca": {
                "components": int(layer_n_components),
                "explained_variance_ratio": layer_pca.explained_variance_ratio_.astype(float).tolist(),
                "explained_variance_cumulative": np.cumsum(layer_pca.explained_variance_ratio_)
                .astype(float)
                .tolist(),
            },
        },
    }


def _plot_progression(
    truthful: np.ndarray,
    untruthful: np.ndarray,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    layers = np.arange(truthful.shape[1])
    t_mean = truthful.mean(axis=0)
    t_std = truthful.std(axis=0)
    u_mean = untruthful.mean(axis=0)
    u_std = untruthful.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, t_mean, label="truthful", linewidth=2)
    ax.fill_between(layers, t_mean - t_std, t_mean + t_std, alpha=0.2)
    ax.plot(layers, u_mean, label="untruthful", linewidth=2)
    ax.fill_between(layers, u_mean - u_std, u_mean + u_std, alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_layer_cosine_heatmap(
    cosine_matrix: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cosine_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_analysis(
    data_root: str,
    output_dir: str,
    max_prompts: int,
    temperature: float,
    random_seed: int,
    token_index: int,
    input_token_index: int | None,
) -> None:
    log.info("=" * 72)
    log.info("Token layer progression analysis")
    log.info("  data root      : %s", Path(data_root).resolve())
    log.info("  output dir     : %s", Path(output_dir).resolve())
    log.info("  max prompts    : %d", max_prompts)
    log.info("  temperature    : %.3f", temperature)
    log.info("  random seed    : %d", random_seed)
    log.info("  token index    : %d", token_index)
    log.info(
        "  input token idx: %s",
        "none" if input_token_index is None else str(input_token_index),
    )
    log.info("  train / test   : 80 / 20")
    log.info("=" * 72)

    log.info("Streaming up to %d matching records from %s ...", max_prompts, data_root)
    selected = _load_limited_records(data_root, max_prompts=max_prompts, temperature=temperature)
    log.info("Selected %d records at temperature=%.3f.", len(selected), temperature)

    sample_record = selected[0]
    sample_recorded_tokens = _flatten_recorded_token_activations(sample_record).shape[0]
    sample_input_tokens = (
        sample_record.prompt_hl.shape[1]
        if sample_record.prompt_hl is not None
        else len(sample_record.input_token_ids) if sample_record.input_token_ids is not None else 0
    )
    _, resolved_token_index, token_description, token_label, token_source = _resolve_token_selection(
        sample_record,
        token_index=token_index,
        input_token_index=input_token_index,
    )
    token_slug = token_description

    if token_source == "input":
        log.info(
            "Extracting input-token activations across layers: requested=%d resolved=%d (%s) "
            "[truncated prompt tokens only]",
            input_token_index,
            resolved_token_index,
            token_description,
        )
    else:
        log.info(
            "Extracting recorded token-step activations across layers: requested=%d resolved=%d (%s) "
            "[0=last prompt token, 1=first generated token]",
            token_index,
            resolved_token_index,
            token_description,
        )
    activations = np.stack(
        [
            _resolve_token_selection(
                record,
                token_index=token_index,
                input_token_index=input_token_index,
            )[0]
            for record in selected
        ],
        axis=0,
    )
    y = np.asarray([record.label for record in selected], dtype=np.int64)
    prompt_ids = [record.prompt_id for record in selected]
    log.info(
        "Activation tensor shape: %s | truthful=%d untruthful=%d",
        activations.shape,
        int(y.sum()),
        int((y == 0).sum()),
    )

    metrics = {
        name: np.stack([_layer_metrics(act)[name] for act in activations], axis=0)
        for name in ("l2_norm", "mean_abs", "signed_mean")
    }
    orthogonality = _orthogonality_metrics(activations)

    truthful_mask = y == 1
    untruthful_mask = y == 0
    if truthful_mask.sum() == 0 or untruthful_mask.sum() == 0:
        raise RuntimeError("Need at least one truthful and one untruthful prompt.")

    base_out_dir = Path(output_dir)
    out_dir = base_out_dir / token_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "layer_metrics.npz"
    summary_path = out_dir / "analysis_summary.json"
    pca_plot_path = out_dir / "pca_scatter.png"
    pca_progression_path = out_dir / "pca_layer_progression.png"
    orth_truthful_heatmap_path = out_dir / "layer_cosine_truthful_heatmap.png"
    orth_untruthful_heatmap_path = out_dir / "layer_cosine_untruthful_heatmap.png"
    orth_gap_heatmap_path = out_dir / "layer_cosine_gap_heatmap.png"

    log.info("Running PCA + linear probe on flattened %s activations ...", token_label)
    probe_outputs = _run_pca_linear_probe(
        activations=activations,
        y=y,
        random_seed=random_seed,
    )
    probe_summary = probe_outputs["summary"]
    log.info(
        "Probe split sizes: train=%d test=%d | PCA components=%d",
        probe_summary["train_size"],
        probe_summary["test_size"],
        probe_summary["pca_components"],
    )
    log.info(
        "PCA explained variance (first 3 comps): %s",
        [round(v, 4) for v in probe_summary["explained_variance_ratio"][:3]],
    )
    log.info(
        "Layerwise PCA explained variance: %s",
        [round(v, 4) for v in probe_summary["layerwise_pca"]["explained_variance_ratio"][:2]],
    )
    log.info(
        "Probe accuracy=%.4f | f1=%.4f | confusion=%s",
        probe_summary["probe_accuracy"],
        probe_summary["probe_f1"],
        probe_summary["probe_confusion"],
    )

    log.info("Saving per-layer arrays to %s ...", features_path)
    np.savez_compressed(
        features_path,
        activations=activations,
        flat_activations=probe_outputs["flat_activations"],
        y=y,
        prompt_ids=np.asarray(prompt_ids, dtype=object),
        l2_norm=metrics["l2_norm"],
        mean_abs=metrics["mean_abs"],
        signed_mean=metrics["signed_mean"],
        layer_cosine_matrices=orthogonality["cosine_matrices"],
        mean_abs_offdiag_cosine=orthogonality["mean_abs_offdiag_cosine"],
        mean_offdiag_cosine=orthogonality["mean_offdiag_cosine"],
        max_abs_offdiag_cosine=orthogonality["max_abs_offdiag_cosine"],
        train_indices=probe_outputs["train_indices"],
        test_indices=probe_outputs["test_indices"],
        pca_projection=probe_outputs["pca_projection"],
        layerwise_pca_projection=probe_outputs["layerwise_pca_projection"],
        probe_test_pred=probe_outputs["probe_test_pred"],
        probe_test_prob=probe_outputs["probe_test_prob"],
    )

    l2_truthful = metrics["l2_norm"][truthful_mask]
    l2_untruthful = metrics["l2_norm"][untruthful_mask]
    mean_abs_truthful = metrics["mean_abs"][truthful_mask]
    mean_abs_untruthful = metrics["mean_abs"][untruthful_mask]
    signed_truthful = metrics["signed_mean"][truthful_mask]
    signed_untruthful = metrics["signed_mean"][untruthful_mask]
    orth_truthful = orthogonality["cosine_matrices"][truthful_mask]
    orth_untruthful = orthogonality["cosine_matrices"][untruthful_mask]

    l2_gap = l2_truthful.mean(axis=0) - l2_untruthful.mean(axis=0)
    mean_abs_gap = mean_abs_truthful.mean(axis=0) - mean_abs_untruthful.mean(axis=0)
    signed_gap = signed_truthful.mean(axis=0) - signed_untruthful.mean(axis=0)
    orth_truthful_mean = orth_truthful.mean(axis=0)
    orth_untruthful_mean = orth_untruthful.mean(axis=0)
    orth_gap = orth_truthful_mean - orth_untruthful_mean
    orth_off_diag_mask = ~np.eye(orth_truthful_mean.shape[0], dtype=bool)
    orth_gap_masked = np.abs(orth_gap.copy())
    orth_gap_masked[~orth_off_diag_mask] = -1.0
    largest_gap_flat = np.argsort(orth_gap_masked.reshape(-1))[::-1][:10]
    largest_gap_pairs = [
        [int(idx // orth_gap.shape[1]), int(idx % orth_gap.shape[1])] for idx in largest_gap_flat
    ]

    _plot_progression(
        l2_truthful,
        l2_untruthful,
        out_dir / "layer_progression_l2_norm.png",
        title=f"{token_label} activation L2 norm across layers",
        ylabel="L2 norm",
    )
    _plot_progression(
        mean_abs_truthful,
        mean_abs_untruthful,
        out_dir / "layer_progression_mean_abs.png",
        title=f"{token_label} activation mean |value| across layers",
        ylabel="Mean absolute activation",
    )
    _plot_progression(
        signed_truthful,
        signed_untruthful,
        out_dir / "layer_progression_signed_mean.png",
        title=f"{token_label} activation signed mean across layers",
        ylabel="Signed mean activation",
    )
    _plot_pca_scatter(
        pca_projection=probe_outputs["pca_projection"],
        y=y,
        train_indices=probe_outputs["train_indices"],
        test_indices=probe_outputs["test_indices"],
        explained_variance_ratio=np.asarray(probe_summary["explained_variance_ratio"], dtype=float),
        out_path=pca_plot_path,
        token_label=token_label,
    )
    _plot_layerwise_pca_progression(
        layerwise_pca_projection=probe_outputs["layerwise_pca_projection"],
        y=y,
        explained_variance_ratio=np.asarray(
            probe_summary["layerwise_pca"]["explained_variance_ratio"], dtype=float
        ),
        out_path=pca_progression_path,
        token_label=token_label,
    )
    _plot_layer_cosine_heatmap(
        cosine_matrix=orth_truthful_mean,
        out_path=orth_truthful_heatmap_path,
        title=f"{token_label} truthful mean layer cosine similarity",
    )
    _plot_layer_cosine_heatmap(
        cosine_matrix=orth_untruthful_mean,
        out_path=orth_untruthful_heatmap_path,
        title=f"{token_label} untruthful mean layer cosine similarity",
    )
    _plot_layer_cosine_heatmap(
        cosine_matrix=orth_gap,
        out_path=orth_gap_heatmap_path,
        title=f"{token_label} truthful minus untruthful layer cosine gap",
    )

    summary = {
        "data_root": str(Path(data_root).resolve()),
        "output_base_dir": str(base_out_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "token_source": token_source,
        "token_index_requested": int(token_index),
        "token_index_resolved": int(resolved_token_index),
        "input_token_index_requested": (
            None if input_token_index is None else int(input_token_index)
        ),
        "token_description": token_description,
        "recorded_token_semantics": {
            "0": "last_prompt_token",
            "1": "first_generated_token",
        },
        "num_input_tokens_per_prompt": int(sample_input_tokens),
        "temperature": float(temperature),
        "random_seed": int(random_seed),
        "num_prompts": int(len(selected)),
        "num_truthful": int(y.sum()),
        "num_untruthful": int((y == 0).sum()),
        "num_layers": int(activations.shape[1]),
        "hidden_dim": int(activations.shape[2]),
        "num_recorded_tokens_per_prompt": int(sample_recorded_tokens),
        "probe": probe_summary,
        "metrics": {
            "l2_norm": {
                "truthful_mean": l2_truthful.mean(axis=0).astype(float).tolist(),
                "untruthful_mean": l2_untruthful.mean(axis=0).astype(float).tolist(),
                "mean_gap_truthful_minus_untruthful": l2_gap.astype(float).tolist(),
                "largest_gap_layers": np.argsort(np.abs(l2_gap))[::-1][:10].astype(int).tolist(),
            },
            "mean_abs": {
                "truthful_mean": mean_abs_truthful.mean(axis=0).astype(float).tolist(),
                "untruthful_mean": mean_abs_untruthful.mean(axis=0).astype(float).tolist(),
                "mean_gap_truthful_minus_untruthful": mean_abs_gap.astype(float).tolist(),
                "largest_gap_layers": np.argsort(np.abs(mean_abs_gap))[::-1][:10].astype(int).tolist(),
            },
            "signed_mean": {
                "truthful_mean": signed_truthful.mean(axis=0).astype(float).tolist(),
                "untruthful_mean": signed_untruthful.mean(axis=0).astype(float).tolist(),
                "mean_gap_truthful_minus_untruthful": signed_gap.astype(float).tolist(),
                "largest_gap_layers": np.argsort(np.abs(signed_gap))[::-1][:10].astype(int).tolist(),
            },
            "layer_orthogonality": {
                "metric": "pairwise cosine similarity between layer activations for the selected token",
                "truthful_mean_cosine_matrix": orth_truthful_mean.astype(float).tolist(),
                "untruthful_mean_cosine_matrix": orth_untruthful_mean.astype(float).tolist(),
                "mean_gap_truthful_minus_untruthful": orth_gap.astype(float).tolist(),
                "truthful_mean_abs_offdiag_cosine": float(
                    orthogonality["mean_abs_offdiag_cosine"][truthful_mask].mean()
                ),
                "untruthful_mean_abs_offdiag_cosine": float(
                    orthogonality["mean_abs_offdiag_cosine"][untruthful_mask].mean()
                ),
                "truthful_mean_offdiag_cosine": float(
                    orthogonality["mean_offdiag_cosine"][truthful_mask].mean()
                ),
                "untruthful_mean_offdiag_cosine": float(
                    orthogonality["mean_offdiag_cosine"][untruthful_mask].mean()
                ),
                "truthful_mean_max_abs_offdiag_cosine": float(
                    orthogonality["max_abs_offdiag_cosine"][truthful_mask].mean()
                ),
                "untruthful_mean_max_abs_offdiag_cosine": float(
                    orthogonality["max_abs_offdiag_cosine"][untruthful_mask].mean()
                ),
                "largest_gap_layer_pairs": largest_gap_pairs,
            },
        },
    }
    log.info("Saving summary JSON to %s ...", summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Top layers by truthful-vs-untruthful gap:")
    log.info("  L2 norm      : %s", summary["metrics"]["l2_norm"]["largest_gap_layers"])
    log.info("  Mean abs     : %s", summary["metrics"]["mean_abs"]["largest_gap_layers"])
    log.info("  Signed mean  : %s", summary["metrics"]["signed_mean"]["largest_gap_layers"])
    log.info(
        "  Orthogonality: mean |cos| off-diagonal truthful=%.4f untruthful=%.4f",
        summary["metrics"]["layer_orthogonality"]["truthful_mean_abs_offdiag_cosine"],
        summary["metrics"]["layer_orthogonality"]["untruthful_mean_abs_offdiag_cosine"],
    )
    log.info(
        "PCA/probe summary: accuracy=%.4f f1=%.4f cumulative_var=%.4f",
        probe_summary["probe_accuracy"],
        probe_summary["probe_f1"],
        probe_summary["explained_variance_cumulative"][-1],
    )

    log.info("Saved per-layer arrays to: %s", features_path)
    log.info("Saved summary to: %s", summary_path)
    log.info("Saved PCA plot to: %s", pca_plot_path)
    log.info("Saved PCA layer progression plot to: %s", pca_progression_path)
    log.info("Saved truthful cosine heatmap to: %s", orth_truthful_heatmap_path)
    log.info("Saved untruthful cosine heatmap to: %s", orth_untruthful_heatmap_path)
    log.info("Saved cosine gap heatmap to: %s", orth_gap_heatmap_path)
    log.info("Saved plots to: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze token activation progression across layers for truthful vs untruthful prompts."
    )
    parser.add_argument("--data-root", default="data", help="Directory containing .npz trajectory files.")
    parser.add_argument(
        "--output-dir",
        default="outputs/last_token_pca",
        help="Directory to save PCA/probe outputs.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=50,
        help="Maximum number of prompts to load and analyze (default: 50).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Only analyze records at this temperature (default: 0.0).",
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
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for the 80/20 split and PCA/probe fitting (default: 0).",
    )
    parser.add_argument(
        "--token-index",
        type=int,
        default=0,
        help=(
            "Recorded token step to analyze. 0=last prompt token from prefill, "
            "1=first generated token, 2=second generated token, ... Negative "
            "indices count backward from the end of the recorded trajectory. "
            "Ignored if --input-token-index is provided."
        ),
    )
    parser.add_argument(
        "--input-token-index",
        type=int,
        default=None,
        help=(
            "Prompt/input token index to analyze from the prefill pass. Uses the "
            "truncated prompt tokens actually fed to the model. 0=first input token, "
            "-1=last input token. Requires .npz files containing 'prompt_hl'; older "
            "datasets only support the last input token."
        ),
    )
    args = parser.parse_args()

    logger = _setup_logging(log_file=args.log_file)
    logger.setLevel(getattr(logging, args.log_level))

    run_analysis(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
        temperature=args.temperature,
        random_seed=args.random_seed,
        token_index=args.token_index,
        input_token_index=args.input_token_index,
    )


if __name__ == "__main__":
    main()
