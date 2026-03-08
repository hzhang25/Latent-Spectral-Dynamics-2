from __future__ import annotations

import os

# Thread caps must be set before torch/safetensors initialise their pools.
# This system has a 300 MB stack per thread; spawning hundreds of threads
# exhausts virtual memory and causes pthread_create to fail with EAGAIN.
# 8 threads for compute is more than sufficient for a single-GPU inference job.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "4"   # safetensors: 1 thread per shard file
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# HF cache lives on eagle so all nodes (login and compute) share one copy.
os.environ.setdefault("HF_HOME", "/eagle/GeoSolv/qinan/LSD/hf_cache")
# Disable HuggingFace's XetHub storage backend — it panics on this platform.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PromptKind = Literal["truthful", "untruthful"]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    category: str
    topic: str
    prompt_text: str
    label: int
    kind: PromptKind


@dataclass(frozen=True)
class TrajectoryRecord:
    hk: np.ndarray
    hl: np.ndarray
    logits: np.ndarray
    generated_token_ids: np.ndarray
    generated_text: str
    generated_text_raw: str
    input_token_ids: np.ndarray
    input_tok_len_raw: int
    prompt_hl: np.ndarray
    prompt_logits: np.ndarray
    reasoning_trace_found: bool
    reasoning_text: str
    reasoning_token_ids: np.ndarray
    reasoning_token_mask: np.ndarray
    response_text: str
    kv_cache: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract activations/logits into .npz trajectory files.")
    parser.add_argument(
        "--prompts-file",
        default="experiment/data_prompts.json",
        help="JSON prompt file with id/category/topic/truthful_prompt/untruthful_prompt.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory prefix. Temperature-specific folders are created as data_<temp> siblings.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HF model name/path.")
    parser.add_argument("--t0", type=int, default=20, help="Max prompt tokens (generation block size).")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Generated blocks per sample. New tokens = num_blocks * t0.",
    )
    parser.add_argument("--deterministic-temp", type=float, default=0.0, help="Deterministic generation temperature.")
    parser.add_argument("--mid-temp", type=float, default=0.3, help="Intermediate stochastic temperature.")
    parser.add_argument("--stochastic-temp", type=float, default=0.7, help="Stochastic generation temperature.")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Explicit list of temperatures to extract. Defaults to deterministic/mid/stochastic temps.",
    )
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
    parser.add_argument(
        "--save-kv-cache",
        action="store_true",
        help=(
            "Save the final KV-cache state into each .npz as 'kv_cache' "
            "[num_layers, 2, n_kv_heads, total_seq_len, head_dim] float32. "
            "~18-36 MB per file for LLaMA-3.1-8B; use carefully with many prompts."
        ),
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
    # Kept for API compatibility; padding is no longer used.
    return ids[-target_len:] if len(ids) > target_len else ids


def topk_logits_values(logits: torch.Tensor, k: int) -> np.ndarray:
    k = min(k, int(logits.shape[-1]))
    values, _ = torch.topk(logits, k=k, dim=-1)
    return values.detach().cpu().float().numpy()


def format_temperature_suffix(temperature: float) -> str:
    if float(temperature).is_integer():
        return str(int(temperature))
    return format(float(temperature), "g")


def temperature_output_dir(base_output_dir: Path, temperature: float) -> Path:
    base_name = base_output_dir.name or "data"
    parent = base_output_dir.parent if base_output_dir.name else base_output_dir
    return parent / f"{base_name}_{format_temperature_suffix(temperature)}"


def extract_reasoning_trace(
    tokenizer: AutoTokenizer,
    generated_token_ids: Sequence[int],
) -> tuple[str, str, str, np.ndarray, np.ndarray, bool]:
    if not generated_token_ids:
        empty_ids = np.asarray([], dtype=np.int32)
        empty_mask = np.asarray([], dtype=bool)
        return "", "", "", empty_ids, empty_mask, False

    token_pieces = [
        tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in generated_token_ids
    ]
    raw_text = "".join(token_pieces)
    token_offsets: List[tuple[int, int]] = []
    cursor = 0
    for piece in token_pieces:
        next_cursor = cursor + len(piece)
        token_offsets.append((cursor, next_cursor))
        cursor = next_cursor

    patterns = [
        r"<think>(.*?)</think>",
        r"<reasoning>(.*?)</reasoning>",
        r"<\|think\|>(.*?)<\|/think\|>",
    ]
    reasoning_spans: List[tuple[int, int]] = []
    reasoning_chunks: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL):
            content_start, content_end = match.span(1)
            if content_end > content_start:
                reasoning_spans.append((content_start, content_end))
                reasoning_chunks.append(match.group(1).strip())

    open_ended_patterns = [
        r"<think>(.*)$",
        r"<reasoning>(.*)$",
        r"<\|think\|>(.*)$",
    ]
    if not reasoning_spans:
        for pattern in open_ended_patterns:
            for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL):
                content_start, content_end = match.span(1)
                if content_end > content_start:
                    reasoning_spans.append((content_start, content_end))
                    reasoning_chunks.append(match.group(1).strip())

    if not reasoning_spans:
        empty_ids = np.asarray([], dtype=np.int32)
        empty_mask = np.zeros(len(generated_token_ids), dtype=bool)
        return raw_text, raw_text.strip(), "", empty_ids, empty_mask, False

    reasoning_mask = np.zeros(len(generated_token_ids), dtype=bool)
    for token_idx, (token_start, token_end) in enumerate(token_offsets):
        if any(token_start < span_end and span_start < token_end for span_start, span_end in reasoning_spans):
            reasoning_mask[token_idx] = True

    reasoning_token_ids = np.asarray(generated_token_ids, dtype=np.int32)[reasoning_mask]
    response_text = raw_text
    for span_pattern in patterns:
        response_text = re.sub(span_pattern, "", response_text, flags=re.IGNORECASE | re.DOTALL)
    for span_pattern in open_ended_patterns:
        response_text = re.sub(span_pattern, "", response_text, flags=re.IGNORECASE | re.DOTALL)
    return (
        raw_text,
        response_text.strip(),
        "\n\n".join(chunk for chunk in reasoning_chunks if chunk).strip(),
        reasoning_token_ids,
        reasoning_mask,
        True,
    )


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
    save_kv_cache: bool = False,
) -> TrajectoryRecord:
    input_ids_raw = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not input_ids_raw:
        raise RuntimeError("Prompt tokenization produced an empty input sequence.")
    # Keep the entire prompt for prefill so prompt-side activations cover every
    # input token, not just the last t0-token window used for generation blocks.
    input_ids = input_ids_raw
    input_tok_len_raw = len(input_ids_raw)

    total_steps = num_blocks * t0
    per_step_layers: List[np.ndarray] = []
    per_step_logits: List[np.ndarray] = []
    generated_token_ids: List[int] = []
    prompt_layers: np.ndarray | None = None
    prompt_logits: np.ndarray | None = None

    # Prefill: run the full prompt once and seed the KV cache.
    # Subsequent steps only pass the single new token, reusing cached keys/values.
    # This avoids re-processing the full growing sequence every step (~50-100x speedup).
    current_input = torch.tensor([input_ids], dtype=torch.long, device=device)
    past_key_values = None

    for step_idx in range(total_steps):
        outputs = model(
            current_input,
            output_hidden_states=True,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")

        if step_idx == 0:
            # Save the full prompt-token hidden states from the prefill pass so
            # downstream analysis can select arbitrary prompt-token positions.
            prompt_layers = np.stack(
                [
                    hidden_states[layer_idx][0, :, :].detach().cpu().float().numpy()
                    for layer_idx in range(1, len(hidden_states))
                ],
                axis=0,
            )  # [L, prompt_len, d]
            prompt_logits = topk_logits_values(outputs.logits[0, :, :], k=logit_topk)  # [prompt_len, topk]

        # hidden_states[0] is embeddings; layers are 1..L
        layer_vecs = [
            hidden_states[layer_idx][0, -1, :].detach().cpu().float().numpy()
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
        generated_token_ids.append(int(next_token.item()))
        # After the prefill step, only feed the newly generated token
        current_input = next_token.view(1, 1)

    steps_layers = np.stack(per_step_layers, axis=0)  # [steps, L, d]
    steps_logits = np.stack(per_step_logits, axis=0)  # [steps, topk]
    hl = np.transpose(steps_layers, (1, 0, 2))  # [L, T, d]

    hk = steps_layers.reshape(num_blocks, t0, steps_layers.shape[1], steps_layers.shape[2])
    hk = np.transpose(hk, (0, 2, 1, 3))  # [num_blocks, L, T0, d]
    logits = steps_logits.reshape(num_blocks, t0, steps_logits.shape[1])  # [num_blocks, T0, topk]
    generated_token_ids_flat = np.asarray(generated_token_ids, dtype=np.int32)
    generated_token_ids_array = generated_token_ids_flat.reshape(num_blocks, t0)
    input_token_ids_array = np.asarray(input_ids, dtype=np.int32)
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    if prompt_layers is None or prompt_logits is None:
        raise RuntimeError("Failed to capture prompt activations from the prefill pass.")

    generated_text_raw, response_text, reasoning_text, reasoning_token_ids, reasoning_token_mask, reasoning_trace_found = (
        extract_reasoning_trace(tokenizer, generated_token_ids)
    )

    # KV cache: transformers 5.x returns a DynamicCache with .key_cache / .value_cache
    # lists; older versions return a tuple of (key, value) per layer.
    # Output shape: [num_layers, 2, n_kv_heads, total_seq_len, head_dim] (batch squeezed).
    kv_cache_array: np.ndarray | None = None
    if save_kv_cache and past_key_values is not None:
        kv_layers = []
        if hasattr(past_key_values, "key_cache"):
            for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
                k = k_tensor.squeeze(0).detach().cpu().float().numpy()
                v = v_tensor.squeeze(0).detach().cpu().float().numpy()
                kv_layers.append(np.stack([k, v], axis=0))
        else:
            for layer_kv in past_key_values:
                k = layer_kv[0].squeeze(0).detach().cpu().float().numpy()
                v = layer_kv[1].squeeze(0).detach().cpu().float().numpy()
                kv_layers.append(np.stack([k, v], axis=0))
        kv_cache_array = np.stack(kv_layers, axis=0)

    return TrajectoryRecord(
        hk=hk,
        hl=hl,
        logits=logits,
        generated_token_ids=generated_token_ids_array,
        generated_text=generated_text,
        generated_text_raw=generated_text_raw,
        input_token_ids=input_token_ids_array,
        input_tok_len_raw=input_tok_len_raw,
        prompt_hl=prompt_layers,
        prompt_logits=prompt_logits,
        reasoning_trace_found=reasoning_trace_found,
        reasoning_text=reasoning_text,
        reasoning_token_ids=reasoning_token_ids,
        reasoning_token_mask=reasoning_token_mask,
        response_text=response_text,
        kv_cache=kv_cache_array,
    )


def write_record(
    output_dir: Path,
    prompt_item: PromptItem,
    temperature: float,
    record: TrajectoryRecord,
) -> None:
    file_name = f"{prompt_item.prompt_id}_temp_{temperature:.1f}.npz"
    out_path = output_dir / file_name
    payload = {
        "hk": record.hk,
        "hl": record.hl,
        "logits": record.logits,
        "label": np.array(prompt_item.label),
        "prompt_id": np.array(prompt_item.prompt_id),
        "temperature": np.array(float(temperature)),
        "category": np.array(prompt_item.category),
        "topic": np.array(prompt_item.topic),
        "prompt_text": np.array(prompt_item.prompt_text),
        "prompt_kind": np.array(prompt_item.kind),
        "generated_token_ids": record.generated_token_ids,
        "generated_text": np.array(record.generated_text),
        "generated_text_raw": np.array(record.generated_text_raw),
        "input_token_ids": record.input_token_ids,
        "input_tok_len_raw": np.array(record.input_tok_len_raw),
        "prompt_hl": record.prompt_hl,
        "prompt_logits": record.prompt_logits,
        "reasoning_trace_found": np.array(record.reasoning_trace_found),
        "reasoning_text": np.array(record.reasoning_text),
        "reasoning_token_ids": record.reasoning_token_ids,
        "reasoning_token_mask": record.reasoning_token_mask,
        "response_text": np.array(record.response_text),
    }
    # kv_cache shape: [num_layers, 2, n_kv_heads, total_seq_len, head_dim]
    if record.kv_cache is not None:
        payload["kv_cache"] = record.kv_cache
    np.savez_compressed(out_path, **payload)


def prompt_to_text(prompt_item: PromptItem, tokenizer=None) -> str:
    user_content = f"Topic: {prompt_item.topic}\n{prompt_item.prompt_text}"
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_content


def resolve_model(model_name: str) -> str:
    """Return the local snapshot path for *model_name*.

    If the snapshot is fully cached it is returned immediately (offline-safe).
    A partial cache (metadata present but weight files missing) is treated as a
    cache miss.  Download uses the ALCF outbound proxy and works from login
    nodes; compute nodes always hit the completed cache.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError, EntryNotFoundError

    # Fast path: metadata AND weight files present — no network needed.
    try:
        path = snapshot_download(model_name, local_files_only=True)
        snap = Path(path)
        if any(snap.glob("*.safetensors")) or any(snap.glob("*.bin")):
            return path
        print(f"[resolve_model] Snapshot at {path} has no weight files — will re-download.")
    except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
        pass

    # Slow path: download via ALCF outbound proxy.
    print(f"[resolve_model] Downloading '{model_name}' via ALCF proxy (login node only)...")
    os.environ.setdefault("https_proxy", "http://proxy.alcf.anl.gov:3128")
    os.environ.setdefault("http_proxy",  "http://proxy.alcf.anl.gov:3128")
    return snapshot_download(model_name)


def main() -> None:
    args = parse_args()
    prompts_file = Path(args.prompts_file)
    output_dir = Path(args.output_dir)

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

    if args.temperatures is None:
        temperatures = [args.deterministic_temp, args.mid_temp, args.stochastic_temp]
    else:
        temperatures = list(args.temperatures)
    temperatures = list(dict.fromkeys(float(temp) for temp in temperatures))

    for temperature in temperatures:
        temperature_output_dir(output_dir, temperature).mkdir(parents=True, exist_ok=True)

    # Partition items across MPI ranks so each GPU processes a unique subset.
    rank = int(os.environ.get("PMI_RANK", 0))
    world_size = int(os.environ.get("PMI_SIZE", 1))
    items = items[rank::world_size]
    if not items:
        print(f"Rank {rank}: no items assigned, exiting.")
        return

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    print(f"Rank {rank}/{world_size} | items {len(items)} | "
          f"[device] {device} | [dtype] {dtype} | "
          f"[CUDA_VISIBLE_DEVICES] {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    local_model_path = resolve_model(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load weights directly onto the target device by setting it as the default
    # allocation device for the duration of from_pretrained. This avoids the
    # CPU staging + separate model.to(device) transfer entirely.
    print(f"Loading model directly onto {device}...")
    with torch.device(device):
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            dtype=dtype,
        )
    model.eval()
    print(f"Model ready on {next(model.parameters()).device}.")

    total = len(items)
    for idx, item in enumerate(items):
        text = prompt_to_text(item, tokenizer)
        print(f"[rank {rank}] {idx + 1}/{total}  {item.prompt_id}", flush=True)

        for temperature in temperatures:
            record = generate_trajectory(
                model=model,
                tokenizer=tokenizer,
                text=text,
                t0=args.t0,
                num_blocks=args.num_blocks,
                temperature=temperature,
                logit_topk=args.logit_topk,
                device=device,
                save_kv_cache=args.save_kv_cache,
            )
            write_record(
                output_dir=temperature_output_dir(output_dir, temperature),
                prompt_item=item,
                temperature=temperature,
                record=record,
            )
        print(f"[rank {rank}] {idx + 1}/{total}  {item.prompt_id}  DONE", flush=True)

    print(f"[rank {rank}] All {total} items complete. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

