"""
inspect_outputs.py  --  print a report of .npz files in a data directory

Usage:
    python3 inspect_outputs.py
    python3 inspect_outputs.py --data-dir data_0 --max-files 10 --max-text 120
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _str(arr) -> str:
    return str(arr.item()) if arr.ndim == 0 else str(arr)


def _flag_issues(rec: dict, text: str) -> list[str]:
    issues = []
    if len(text.strip()) == 0:
        issues.append("EMPTY_TEXT")
    elif len(text.strip()) < 20:
        issues.append("VERY_SHORT")
    words = text.split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) != len(set(trigrams)):
            counts = {t: trigrams.count(t) for t in set(trigrams)}
            max_rep = max(counts.values())
            if max_rep >= 3:
                issues.append(f"REPETITION(x{max_rep})")
    if "generated_token_ids" in rec:
        ids = rec["generated_token_ids"].flatten()
        if len(ids) > 0 and np.all(ids == ids[0]):
            issues.append("ALL_SAME_TOKEN")
    return issues

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_0", help="Directory with .npz files.")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of .npz files to inspect (0 = all).")
    parser.add_argument("--max-text", type=int, default=0, help="Max chars of generated text to display (0 = unlimited).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.npz"))
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print(f"No .npz files found in {data_dir.resolve()}")
        return

    records = []
    shape_errors = []

    for path in files:
        try:
            rec = _load(path)
        except Exception as e:
            print(f"[ERROR] Could not load {path.name}: {e}")
            continue

        text        = _str(rec["generated_text"]) if "generated_text" in rec else "<missing>"
        prompt_text = _str(rec["prompt_text"])    if "prompt_text"  in rec else "<missing>"
        prompt_id   = _str(rec["prompt_id"])      if "prompt_id"  in rec else path.stem
        kind        = _str(rec["prompt_kind"])     if "prompt_kind" in rec else "?"
        label       = int(rec["label"].item())     if "label"      in rec else -1
        temp        = float(rec["temperature"].item()) if "temperature" in rec else float("nan")
        topic       = _str(rec["topic"])           if "topic"      in rec else "?"
        category    = _str(rec["category"])        if "category"   in rec else "?"

        # Output token count
        gen_len = int(rec["generated_token_ids"].size) if "generated_token_ids" in rec else 0

        # Input token counts
        # input_tok_len_raw = true prompt length before pad/truncate (new field)
        # in_tok_padded     = t0, derived from hk shape [num_blocks, L, T0, d]
        in_tok_raw    = int(rec["input_tok_len_raw"].item()) if "input_tok_len_raw" in rec else None
        in_tok_padded = int(rec["hk"].shape[2])             if "hk" in rec else None

        hk_shape = tuple(rec["hk"].shape) if "hk" in rec else None
        hl_shape = tuple(rec["hl"].shape) if "hl" in rec else None
        lo_shape = tuple(rec["logits"].shape) if "logits" in rec else None
        # kv_cache: [num_layers, 2, n_kv_heads, total_seq_len, head_dim]
        kv_shape = tuple(rec["kv_cache"].shape) if "kv_cache" in rec else None
        kv_mb    = round(rec["kv_cache"].nbytes / 1e6, 1) if "kv_cache" in rec else None

        issues = _flag_issues(rec, text)
        if in_tok_raw is not None and in_tok_padded is not None:
            if in_tok_raw > in_tok_padded:
                issues.append(f"TRUNCATED({in_tok_raw}->{in_tok_padded})")

        records.append(dict(
            file=path.name,
            prompt_id=prompt_id,
            kind=kind,
            label=label,
            temp=temp,
            topic=topic,
            category=category,
            prompt_text=prompt_text,
            text=text,
            in_tok_raw=in_tok_raw,
            in_tok_padded=in_tok_padded,
            gen_len=gen_len,
            hk_shape=hk_shape,
            hl_shape=hl_shape,
            lo_shape=lo_shape,
            kv_shape=kv_shape,
            kv_mb=kv_mb,
            issues=issues,
        ))

    # ── summary stats ────────────────────────────────────────────────────────
    n = len(records)
    n_truthful   = sum(1 for r in records if r["kind"] == "truthful")
    n_untruthful = sum(1 for r in records if r["kind"] == "untruthful")
    n_with_text  = sum(1 for r in records if r["text"] != "<missing>")
    avg_out_len  = np.mean([r["gen_len"] for r in records]) if records else 0
    raw_lens     = [r["in_tok_raw"] for r in records if r["in_tok_raw"] is not None]
    avg_in_raw   = np.mean(raw_lens) if raw_lens else None
    n_truncated  = sum(1 for r in records if r["in_tok_raw"] is not None and r["in_tok_padded"] is not None and r["in_tok_raw"] > r["in_tok_padded"])
    n_issues     = sum(1 for r in records if r["issues"])

    hk_shapes  = {r["hk_shape"] for r in records if r["hk_shape"]}
    hl_shapes  = {r["hl_shape"] for r in records if r["hl_shape"]}
    lo_shapes  = {r["lo_shape"] for r in records if r["lo_shape"]}

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Files analyzed      : {n}")
    print(f"  Truthful samples    : {n_truthful}")
    print(f"  Untruthful samples  : {n_untruthful}")
    print(f"  Files with text     : {n_with_text}/{n}")
    print(f"  Avg input tokens    : {avg_in_raw:.1f}" if avg_in_raw is not None else "  Avg input tokens    : n/a (old files)")
    print(f"  Input truncated (raw>t0): {n_truncated}")
    print(f"  Avg output tokens   : {avg_out_len:.1f}")
    print(f"  Files with issues   : {n_issues}")
    print(f"  hk shapes seen      : {hk_shapes}")
    print(f"  hl shapes seen      : {hl_shapes}")
    print(f"  logits shapes seen  : {lo_shapes}")
    kv_shapes = {r["kv_shape"] for r in records if r["kv_shape"]}
    n_with_kv = sum(1 for r in records if r["kv_shape"])
    total_kv_mb = sum(r["kv_mb"] for r in records if r["kv_mb"])
    if n_with_kv:
        avg_kv_mb = total_kv_mb / n_with_kv
        print(f"  kv_cache shapes     : {kv_shapes}")
        print(f"  Files with kv_cache : {n_with_kv}/{n}  (avg {avg_kv_mb:.1f} MB, total {total_kv_mb:.0f} MB)")
    else:
        print(f"  kv_cache            : not saved (use --save-kv-cache to enable)")

    # ── per-file table ────────────────────────────────────────────────────────
    col = "{:<38} {:<12} {:<10} {:>7} {:>7} {:>7}  {}  [{}]"
    print()
    print("=" * 100)
    print("PER-FILE REPORT")
    print("=" * 100)
    print(col.format("prompt_id", "kind", "topic[:10]", "in_raw", "in_pad", "out", "generated_text", "issues"))
    print("-" * 100)
    for r in records:
        full_text   = r["text"].replace("\n", " ")
        disp_text   = full_text if args.max_text == 0 else full_text[:args.max_text]
        short_topic = r["topic"][:10]
        issue_str   = ", ".join(r["issues"]) if r["issues"] else "ok"
        in_raw = str(r["in_tok_raw"]) if r["in_tok_raw"] is not None else "n/a"
        in_pad = str(r["in_tok_padded"]) if r["in_tok_padded"] is not None else "n/a"
        print(col.format(r["prompt_id"], r["kind"], short_topic,
                          in_raw, in_pad, r["gen_len"], disp_text, issue_str))

    # ── truthful vs untruthful text comparison ────────────────────────────────
    pairs: dict[str, dict] = defaultdict(dict)
    for r in records:
        m = re.match(r"(pair_\d+)_(truthful|untruthful)", r["prompt_id"])
        if m:
            pairs[m.group(1)][m.group(2)] = r

    print()
    print("=" * 80)
    print("TRUTHFUL vs UNTRUTHFUL COMPARISON (up to first 10 pairs in inspected files)")
    print("=" * 80)
    for pair_key in sorted(pairs)[:10]:
        pair = pairs[pair_key]
        topic = pair.get("truthful", pair.get("untruthful", {})).get("topic", "?")

        t_prompt = pair.get("truthful",   {}).get("prompt_text", "<missing>").replace("\n", " ")
        u_prompt = pair.get("untruthful", {}).get("prompt_text", "<missing>").replace("\n", " ")
        t_full   = pair.get("truthful",   {}).get("text", "<missing>").replace("\n", " ")
        u_full   = pair.get("untruthful", {}).get("text", "<missing>").replace("\n", " ")
        if args.max_text:
            t_prompt = t_prompt[:args.max_text]
            u_prompt = u_prompt[:args.max_text]
            t_full   = t_full[:args.max_text]
            u_full   = u_full[:args.max_text]

        t_in  = pair.get("truthful",   {}).get("in_tok_raw", "?")
        u_in  = pair.get("untruthful", {}).get("in_tok_raw", "?")
        t_out = pair.get("truthful",   {}).get("gen_len", "?")
        u_out = pair.get("untruthful", {}).get("gen_len", "?")

        print(f"\n  [{pair_key}]  topic: {topic}")
        print(f"    TRUTHFUL   input  [{t_in} tok]: {t_prompt}")
        print(f"    TRUTHFUL   output [{t_out} tok]: {t_full}")
        print(f"    UNTRUTHFUL input  [{u_in} tok]: {u_prompt}")
        print(f"    UNTRUTHFUL output [{u_out} tok]: {u_full}")

    # ── flagged files ─────────────────────────────────────────────────────────
    flagged = [r for r in records if r["issues"]]
    if flagged:
        print()
        print("=" * 80)
        print(f"FLAGGED FILES ({len(flagged)})")
        print("=" * 80)
        for r in flagged:
            print(f"  {r['prompt_id']:40s}  issues: {', '.join(r['issues'])}")
            print(f"    text: {r['text'].replace(chr(10), ' ')[:120]}")
    else:
        print("\nNo issues flagged.")

    print()
    print(f"Done. {n} files in {data_dir.resolve()}")


if __name__ == "__main__":
    main()
