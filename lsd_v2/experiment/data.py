from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class TrajectoryRecord:
    prompt_id: str
    label: int  # 1 = truthful, 0 = untruthful
    temperature: float
    rollout_id: Optional[int]
    hk: np.ndarray  # [num_blocks, L, T0, d]
    hl: Optional[np.ndarray]  # [L, T, d]
    logits: Optional[np.ndarray]  # [num_blocks, T0, vocab]
    prompt_hl: Optional[np.ndarray] = None  # [L, prompt_len, d] from prefill
    input_token_ids: Optional[np.ndarray] = None  # [prompt_len] after truncation to t0
    input_tok_len_raw: Optional[int] = None  # prompt length before truncation


def _parse_label(raw_label: np.ndarray) -> int:
    if raw_label.dtype.kind in {"i", "u", "f"}:
        return int(raw_label.item())
    text = str(raw_label.item()).strip().lower()
    if text in {"truthful", "true", "1"}:
        return 1
    if text in {"untruthful", "false", "0"}:
        return 0
    raise ValueError(f"Unsupported label value: {text}")


def load_records(
    data_root: str | Path,
    max_pairs: Optional[int] = None,
) -> List[TrajectoryRecord]:
    """
    Expected file format per .npz:
      - required: hk, label, prompt_id, temperature
      - optional: hl, logits, rollout_id

    max_pairs: if set, load at most this many prompt pairs (2 * max_pairs files).
    Files are sorted alphabetically, so pair_NNNN_truthful always precedes
    pair_NNNN_untruthful, keeping pairs intact.
    """
    root = Path(data_root)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {root}")
    if max_pairs is not None:
        files = files[: max_pairs * 2]

    records: List[TrajectoryRecord] = []
    for file in files:
        with np.load(file, allow_pickle=True) as npz:
            hk = np.asarray(npz["hk"])
            hl = np.asarray(npz["hl"]) if "hl" in npz else None
            logits = np.asarray(npz["logits"]) if "logits" in npz else None
            label = _parse_label(npz["label"])
            prompt_id = str(npz["prompt_id"].item())
            temp = float(npz["temperature"].item())
            rollout_id = int(npz["rollout_id"].item()) if "rollout_id" in npz else None

            records.append(
                TrajectoryRecord(
                    prompt_id=prompt_id,
                    label=label,
                    temperature=temp,
                    rollout_id=rollout_id,
                    hk=hk,
                    hl=hl,
                    logits=logits,
                    prompt_hl=np.asarray(npz["prompt_hl"]) if "prompt_hl" in npz else None,
                    input_token_ids=np.asarray(npz["input_token_ids"]) if "input_token_ids" in npz else None,
                    input_tok_len_raw=(
                        int(npz["input_tok_len_raw"].item()) if "input_tok_len_raw" in npz else None
                    ),
                )
            )
    return records


def split_by_regime(
    records: Iterable[TrajectoryRecord], deterministic_temp: float, stochastic_temp: float
) -> Dict[str, List[TrajectoryRecord]]:
    deterministic = [r for r in records if np.isclose(r.temperature, deterministic_temp)]
    stochastic = [r for r in records if np.isclose(r.temperature, stochastic_temp)]
    return {"deterministic": deterministic, "stochastic": stochastic}


def group_by_prompt(records: Iterable[TrajectoryRecord]) -> Dict[str, List[TrajectoryRecord]]:
    grouped: Dict[str, List[TrajectoryRecord]] = {}
    for record in records:
        grouped.setdefault(record.prompt_id, []).append(record)
    return grouped

