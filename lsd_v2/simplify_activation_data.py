from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np


# Keep only the tensors and metadata needed for activation-focused analysis.
KEEP_KEYS = (
    "hk",
    "prompt_hl",
    "label",
    "prompt_id",
    "temperature",
    "rollout_id",
    "prompt_text",
    "prompt_kind",
    "category",
    "topic",
    "input_token_ids",
    "input_tok_len_raw",
)


TEMPERATURE_SWEEP = (0.0, 0.3, 0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy activation-focused .npz files into a simplified directory without modifying the source data."
        )
    )
    parser.add_argument(
        "--source-dir",
        default="data_0",
        help="Directory containing the original .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_0_simple",
        help="Directory where simplified .npz files will be written.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on the number of files to copy (0 = all).",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed instead of the faster uncompressed np.savez.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite files that already exist in the output directory.",
    )
    parser.add_argument(
        "--include-temperature-siblings",
        action="store_true",
        help=(
            "Also simplify sibling temperature directories, e.g. "
            "data_0 -> data_0.3/data_0.7 and data_0_simple -> data_0.3_simple/data_0.7_simple."
        ),
    )
    return parser.parse_args()


def is_valid_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except Exception:
        return False


def _format_temperature_suffix(temperature: float) -> str:
    if float(temperature).is_integer():
        return str(int(temperature))
    return format(float(temperature), "g")


def _split_temperature_dir_name(name: str) -> tuple[str, str, str] | None:
    import re

    match = re.match(r"^(?P<prefix>.+?)_(?P<temp>\d+(?:\.\d+)?)(?P<suffix>(?:_.+)?)$", name)
    if match is None:
        return None
    return match.group("prefix"), match.group("temp"), match.group("suffix")


def _resolve_dir_pairs(source_dir: Path, output_dir: Path, include_temperature_siblings: bool) -> list[tuple[Path, Path]]:
    if not include_temperature_siblings:
        return [(source_dir, output_dir)]

    source_parts = _split_temperature_dir_name(source_dir.name)
    output_parts = _split_temperature_dir_name(output_dir.name)
    if source_parts is None or output_parts is None:
        raise ValueError(
            "Temperature sibling mode requires names like data_0 and data_0_simple."
        )

    source_prefix, source_temp, source_suffix = source_parts
    output_prefix, output_temp, output_suffix = output_parts
    if source_temp != "0" or output_temp != "0":
        raise ValueError(
            "Temperature sibling mode expects deterministic roots ending in '_0', "
            "for example source=data_0 and output=data_0_simple."
        )

    pairs = []
    for temperature in TEMPERATURE_SWEEP:
        temp_suffix = _format_temperature_suffix(temperature)
        src = source_dir.with_name(f"{source_prefix}_{temp_suffix}{source_suffix}")
        dst = output_dir.with_name(f"{output_prefix}_{temp_suffix}{output_suffix}")
        pairs.append((src, dst))
    return pairs


def simplify_directory(
    source_dir: Path,
    output_dir: Path,
    max_files: int,
    compressed: bool,
    overwrite: bool,
) -> tuple[int, int]:
    files = sorted(source_dir.glob("*.npz"))
    if max_files > 0:
        files = files[: max_files]

    if not files:
        raise FileNotFoundError(f"No .npz files found in {source_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for src_path in files:
        out_path = output_dir / src_path.name
        if out_path.exists() and not overwrite:
            if is_valid_npz(out_path):
                continue
            print(f"[rewrite] {out_path.name}: existing file is not a valid npz archive")
        with np.load(src_path, allow_pickle=True) as npz:
            payload = {}
            missing_required = [key for key in ("hk", "prompt_hl", "label", "prompt_id", "temperature") if key not in npz]
            if missing_required:
                skipped += 1
                print(f"[skip] {src_path.name}: missing required keys {missing_required}")
                continue
            for key in KEEP_KEYS:
                if key in npz:
                    payload[key] = np.asarray(npz[key])

        tmp_path = output_dir / f".{src_path.name}.tmp"
        if compressed:
            np.savez_compressed(tmp_path, **payload)
        else:
            np.savez(tmp_path, **payload)
        tmp_npz_path = tmp_path if tmp_path.suffix == ".npz" else tmp_path.with_suffix(tmp_path.suffix + ".npz")
        os.replace(tmp_npz_path, out_path)
        copied += 1

    print(f"Copied {copied} simplified files to {output_dir.resolve()}")
    if skipped:
        print(f"Skipped {skipped} files with missing required activation fields.")
    return copied, skipped


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    dir_pairs = _resolve_dir_pairs(
        source_dir=source_dir,
        output_dir=output_dir,
        include_temperature_siblings=args.include_temperature_siblings,
    )
    total_copied = 0
    total_skipped = 0
    for src_dir, dst_dir in dir_pairs:
        copied, skipped = simplify_directory(
            source_dir=src_dir,
            output_dir=dst_dir,
            max_files=args.max_files,
            compressed=args.compressed,
            overwrite=args.overwrite,
        )
        total_copied += copied
        total_skipped += skipped
    if len(dir_pairs) > 1:
        print(
            f"Done across {len(dir_pairs)} directories: copied {total_copied} files, skipped {total_skipped}."
        )


if __name__ == "__main__":
    main()
