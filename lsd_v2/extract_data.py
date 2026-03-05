from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PromptKind = Literal["truthful", "untruthful"]
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    category: str
    topic: str
    prompt_text: str
    label: int
    kind: PromptKind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract activations/logits into .npz trajectory files.")
    parser.add_argument(
        "--prompts-file",
        default="experiment/data_prompts.json",
        help="JSON prompt file with id/category/topic/truthful_prompt/untruthful_prompt.",
    )
    parser.add_argument("--output-dir", default="data", help="Output directory for .npz files.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HF model name/path.")
    parser.add_argument("--t0", type=int, default=16, help="Fixed prompt token length.")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=8,
        help="Generated blocks per sample. New tokens = num_blocks * t0.",
    )
    parser.add_argument(
        "--stochastic-rollouts",
        type=int,
        default=32,
        help="Rollout count for stochastic regime (temperature > 0).",
    )
    parser.add_argument("--deterministic-temp", type=float, default=0.0, help="Deterministic generation temperature.")
    parser.add_argument("--stochastic-temp", type=float, default=0.7, help="Stochastic generation temperature.")
    parser.add_argument(
        "--logit-topk",
        type=int,
        default=256,
        help="Store top-k logits per step to reduce disk usage.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of prompt pairs from JSON for test runs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device selection.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp32"],
        default="auto",
        help="Model dtype selection.",
    )
    parser.add_argument(
        "--include-truthful",
        action="store_true",
        help="Include truthful prompts (default: include both truthful and untruthful).",
    )
    parser.add_argument(
        "--include-untruthful",
        action="store_true",
        help="Include untruthful prompts (default: include both truthful and untruthful).",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def load_prompt_items(prompts_file: Path, max_pairs: int | None, include_kinds: set[PromptKind]) -> List[PromptItem]:
    rows = json.loads(prompts_file.read_text(encoding="utf-8"))
    if max_pairs is not None:
        rows = rows[:max_pairs]

    items: List[PromptItem] = []
    for row in rows:
        base_id = int(row["id"])
        category = str(row["category"])
        topic = str(row["topic"])
        if "truthful" in include_kinds:
            items.append(
                PromptItem(
                    prompt_id=f"pair_{base_id:04d}_truthful",
                    category=category,
                    topic=topic,
                    prompt_text=str(row["truthful_prompt"]),
                    label=1,
                    kind="truthful",
                )
            )
        if "untruthful" in include_kinds:
            items.append(
                PromptItem(
                    prompt_id=f"pair_{base_id:04d}_untruthful",
                    category=category,
                    topic=topic,
                    prompt_text=str(row["untruthful_prompt"]),
                    label=0,
                    kind="untruthful",
                )
            )
    return items


def ensure_exact_length(ids: List[int], target_len: int, pad_id: int) -> List[int]:
    if len(ids) >= target_len:
        return ids[:target_len]
    return ids + [pad_id] * (target_len - len(ids))


def topk_logits_values(logits: torch.Tensor, k: int) -> np.ndarray:
    k = min(k, int(logits.shape[-1]))
    values, _ = torch.topk(logits, k=k, dim=-1)
    return values.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def generate_trajectory(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    t0: int,
    num_blocks: int,
    temperature: float,
    logit_topk: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer must define pad_token_id or eos_token_id.")
    input_ids = ensure_exact_length(input_ids, target_len=t0, pad_id=pad_id)

    running_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    total_steps = num_blocks * t0
    per_step_layers: List[np.ndarray] = []
    per_step_logits: List[np.ndarray] = []

    for _ in range(total_steps):
        outputs = model(running_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")

        # hidden_states[0] is embeddings; layers are 1..L
        layer_vecs = [
            hidden_states[layer_idx][0, -1, :].detach().cpu().numpy().astype(np.float32)
            for layer_idx in range(1, len(hidden_states))
        ]
        per_step_layers.append(np.stack(layer_vecs, axis=0))  # [L, d]

        logits = outputs.logits[0, -1, :]
        per_step_logits.append(topk_logits_values(logits, k=logit_topk))

        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        running_ids = torch.cat([running_ids, next_token.view(1, 1)], dim=1)

    steps_layers = np.stack(per_step_layers, axis=0)  # [steps, L, d]
    steps_logits = np.stack(per_step_logits, axis=0)  # [steps, topk]
    hl = np.transpose(steps_layers, (1, 0, 2))  # [L, T, d]

    hk = steps_layers.reshape(num_blocks, t0, steps_layers.shape[1], steps_layers.shape[2])
    hk = np.transpose(hk, (0, 2, 1, 3))  # [num_blocks, L, T0, d]
    logits = steps_logits.reshape(num_blocks, t0, steps_logits.shape[1])  # [num_blocks, T0, topk]
    return hk, hl, logits


def write_record(
    output_dir: Path,
    prompt_item: PromptItem,
    temperature: float,
    rollout_id: int | None,
    hk: np.ndarray,
    hl: np.ndarray,
    logits: np.ndarray,
) -> None:
    rollout_tag = "det" if rollout_id is None else f"rollout_{rollout_id:02d}"
    file_name = f"{prompt_item.prompt_id}_temp_{temperature:.1f}_{rollout_tag}.npz"
    out_path = output_dir / file_name
    payload = {
        "hk": hk,
        "hl": hl,
        "logits": logits,
        "label": np.array(prompt_item.label),
        "prompt_id": np.array(prompt_item.prompt_id),
        "temperature": np.array(float(temperature)),
        "category": np.array(prompt_item.category),
        "topic": np.array(prompt_item.topic),
        "prompt_kind": np.array(prompt_item.kind),
    }
    if rollout_id is not None:
        payload["rollout_id"] = np.array(rollout_id)
    np.savez_compressed(out_path, **payload)


def prompt_to_text(prompt_item: PromptItem) -> str:
    return f"Topic: {prompt_item.topic}\nInstruction: {prompt_item.prompt_text}"


def main() -> None:
    args = parse_args()
    prompts_file = Path(args.prompts_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_truthful = args.include_truthful
    include_untruthful = args.include_untruthful
    if not include_truthful and not include_untruthful:
        include_kinds: set[PromptKind] = {"truthful", "untruthful"}
    else:
        include_kinds = set()
        if include_truthful:
            include_kinds.add("truthful")
        if include_untruthful:
            include_kinds.add("untruthful")

    items = load_prompt_items(prompts_file, args.max_pairs, include_kinds=include_kinds)
    if not items:
        raise RuntimeError("No prompt items selected. Check flags and input JSON.")

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    for item in tqdm(items, desc="Prompt conditions"):
        text = prompt_to_text(item)

        hk, hl, logits = generate_trajectory(
            model=model,
            tokenizer=tokenizer,
            text=text,
            t0=args.t0,
            num_blocks=args.num_blocks,
            temperature=args.deterministic_temp,
            logit_topk=args.logit_topk,
            device=device,
        )
        write_record(
            output_dir=output_dir,
            prompt_item=item,
            temperature=args.deterministic_temp,
            rollout_id=None,
            hk=hk,
            hl=hl,
            logits=logits,
        )

        for rollout_id in range(args.stochastic_rollouts):
            hk, hl, logits = generate_trajectory(
                model=model,
                tokenizer=tokenizer,
                text=text,
                t0=args.t0,
                num_blocks=args.num_blocks,
                temperature=args.stochastic_temp,
                logit_topk=args.logit_topk,
                device=device,
            )
            write_record(
                output_dir=output_dir,
                prompt_item=item,
                temperature=args.stochastic_temp,
                rollout_id=rollout_id,
                hk=hk,
                hl=hl,
                logits=logits,
            )

    print(f"Data extraction complete. Wrote files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

