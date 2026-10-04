"""Build a trajectory-disjoint adjacent-frame manifest from extracted V1 data."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

def split_trajectories(trajectories, seed=42):
    groups = defaultdict(list)
    for name in sorted(trajectories):
        groups[name.split("/")[0]].append(name)
    result = {"train": [], "val": [], "test": []}
    # Stratify when possible, otherwise globally distribute small-scene groups.
    small = []
    rng = random.Random(seed)
    for names in groups.values():
        rng.shuffle(names)
        if len(names) < 3:
            small.extend(names)
            continue
        nval = max(1, round(len(names) * 0.15))
        ntest = max(1, round(len(names) * 0.15))
        result["val"].extend(names[:nval])
        result["test"].extend(names[nval:nval + ntest])
        result["train"].extend(names[nval + ntest:])
    if small:
        rng.shuffle(small)
        if len(small) >= 3:
            nv = max(1, round(len(small) * 0.15))
            nt = max(1, round(len(small) * 0.15))
            result["val"].extend(small[:nv])
            result["test"].extend(small[nv:nv + nt])
            result["train"].extend(small[nv + nt:])
        else:
            result["train"].extend(small)
    return result

def build_manifest(root, seed=42, scenes=None):
    root = Path(root).resolve()
    trajectories = {}
    for rgb_dir in sorted(root.glob("*/Easy/P*/image_left")):
        trajectory = rgb_dir.parent
        scene = trajectory.relative_to(root).parts[0]
        if scenes and scene not in scenes:
            continue
        pairs = []
        flow_dir = trajectory / "flow"
        flows = sorted(flow_dir.glob("*_flow.npy"))
        if not flows:
            raise RuntimeError(f"Missing complete flow data: {trajectory}")
        for flow in flows:
            frame, nxt, _ = flow.stem.split("_")
            if int(nxt) != int(frame) + 1:
                raise ValueError(f"Non-adjacent flow: {flow}")
            files = {"rgb0": rgb_dir / f"{frame}_left.png", "rgb1": rgb_dir / f"{nxt}_left.png", "depth0": trajectory / "depth_left" / f"{frame}_left_depth.npy", "depth1": trajectory / "depth_left" / f"{nxt}_left_depth.npy", "flow": flow, "flow_mask": flow_dir / f"{frame}_{nxt}_mask.npy"}
            missing = [str(p) for p in files.values() if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete pair {trajectory}/{frame}: {missing}")
            pair = {k: p.relative_to(root).as_posix() for k, p in files.items()}
            pair.update(trajectory=trajectory.relative_to(root).as_posix(), frame_id=int(frame))
            pairs.append(pair)
        trajectories[trajectory.relative_to(root).as_posix()] = pairs
    if not trajectories:
        raise RuntimeError(f"No complete trajectories found under {root}")
    partition = split_trajectories(trajectories, seed)
    return {"version": 1, "root": str(root), "seed": seed, "flow_mask_valid_value": 0, "source": "https://huggingface.co/datasets/theairlabcmu/tartanair", "trajectories": partition, "splits": {k: [pair for trajectory in names for pair in trajectories[trajectory]] for k, names in partition.items()}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/tartanair"))
    p.add_argument("--output", type=Path, default=Path("data/tartanair_manifest.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenes", nargs="+")
    args = p.parse_args()
    manifest = build_manifest(args.root, args.seed, args.scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: {"trajectories": len(manifest["trajectories"][k]), "pairs": len(v)} for k, v in manifest["splits"].items()}, indent=2))

if __name__ == "__main__":
    main()
