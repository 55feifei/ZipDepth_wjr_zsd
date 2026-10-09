#!/usr/bin/env python
"""Generate dataset.py-compatible bidirectional GMFlow estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zipdepth.optical_flow import DEFAULT_CHECKPOINT, DEFAULT_GMFLOW_ROOT, GMFlowEstimator

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return [(0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", path.name)]


def discover_sequences(input_root):
    """Accept either a TartanAir tree or a single ordered RGB frame folder."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if input_root.name == "image_left":
        directories = [input_root]
    else:
        directories = sorted(p for p in input_root.rglob("image_left") if p.is_dir())
        if not directories:
            directories = [input_root]
    sequences = []
    for directory in directories:
        frames = sorted((p for p in directory.iterdir()
                         if p.is_file() and p.suffix.lower() in EXTENSIONS), key=natural_key)
        if len(frames) < 2:
            continue
        if len({p.stem for p in frames}) != len(frames):
            raise ValueError(f"Duplicate image stems would overwrite pair outputs: {directory}")
        trajectory = directory.parent if directory.name == "image_left" else directory
        relative = trajectory.relative_to(input_root) if trajectory.is_relative_to(input_root) else Path()
        sequences.append((relative, frames))
    return sequences


def save_pair(output_root, relative, stem, result):
    """Atomic file replacement prevents interrupted writes from leaving bad .npy files."""
    for key, value in result.items():
        directory = output_root / relative / key
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{stem}.npy"
        temporary = directory / f"{stem}.npy.tmp"
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        temporary.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "tartanair")
    parser.add_argument("--output", type=Path,
                        help="Defaults to INPUT's sibling named INPUT_flow")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gmflow-root", type=Path, default=DEFAULT_GMFLOW_ROOT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--inference-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"),
                        help="Resize for inference, then restore original flow shape and pixel units")
    parser.add_argument("--max-pairs", type=int, help="Maximum newly estimated pairs across all sequences")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate pairs with four existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="List pair counts without loading the model")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.inference_size is not None and min(args.inference_size) < 2:
        parser.error("--inference-size dimensions must be >= 2")
    input_root = args.input.resolve()
    output_root = args.output.resolve() if args.output else input_root.with_name(input_root.name + "_flow")
    sequences = discover_sequences(input_root)
    if not sequences:
        parser.error(f"No sequence containing at least two RGB frames in {input_root}")
    total = sum(len(frames) - 1 for _, frames in sequences)
    print(f"Found {len(sequences)} sequences, {total} adjacent pairs; output: {output_root}", flush=True)
    if args.dry_run:
        for relative, frames in sequences:
            print(f"  {relative}: {len(frames) - 1} pairs")
        return 0
    torch.set_num_threads(args.cpu_threads)
    estimator = GMFlowEstimator(args.checkpoint, args.gmflow_root, args.device, args.inference_size)
    print(f"Using {estimator.device}; checkpoint: {args.checkpoint}", flush=True)
    generated = skipped = visited = 0
    started = time.perf_counter()
    for relative, frames in sequences:
        for first, second in zip(frames[:-1], frames[1:]):
            visited += 1
            files = [output_root / relative / key / f"{first.stem}.npy"
                     for key in ("f_flow", "b_flow", "f_mask", "b_mask")]
            if not args.overwrite and all(file.is_file() for file in files):
                skipped += 1
                continue
            with Image.open(first) as image:
                rgb0 = np.asarray(image.convert("RGB"))
            with Image.open(second) as image:
                rgb1 = np.asarray(image.convert("RGB"))
            result = estimator.infer_pair(rgb0, rgb1)
            save_pair(output_root, relative, first.stem, result)
            generated += 1
            print(f"[{visited}/{total}] {relative}/{first.stem} -> {second.stem} "
                  f"({generated} generated, {time.perf_counter() - started:.1f}s)", flush=True)
            if args.max_pairs is not None and generated >= args.max_pairs:
                break
        if args.max_pairs is not None and generated >= args.max_pairs:
            break
    print(json.dumps({"generated": generated, "skipped": skipped,
                      "total_pairs": total, "output": str(output_root),
                      "seconds": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
