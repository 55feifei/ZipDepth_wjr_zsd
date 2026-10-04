"""Strict adjacent-frame loader for official TartanAir V1 data."""
from __future__ import annotations
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def _resize(x, size, mode):
    batched = x.ndim == 4
    result = F.interpolate(x if batched else x[None], size=(size, size), mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
    return result if batched else result[0]

def transform_pair(image, depth, flow, flow_valid, size, flip=False, max_depth=80.0):
    """Square center crop and resize, including vector magnitude and endpoint mask."""
    h, w = depth.shape[-2:]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    crop = (..., slice(top, top + side), slice(left, left + side))
    image = _resize(image[crop], size, "bilinear")
    valid = torch.isfinite(depth) & (depth > 0) & (depth < max_depth)
    inverse = torch.where(valid, 1 / depth.clamp_min(1e-6), 0)
    # Nearest avoids averaging foreground/background depth and invalid pixels.
    inverse = _resize(inverse[crop], size, "nearest")
    valid = _resize(valid[crop].float(), size, "nearest").bool()
    # The validity uses the same bilinear support as vector interpolation: a
    # target pixel is valid only when all contributing source pixels are valid.
    flow_valid = flow_valid & torch.isfinite(flow).all(0, keepdim=True)
    flow = torch.where(flow_valid, flow, torch.zeros_like(flow))
    flow = _resize(flow[crop], size, "bilinear") * (size / side)
    flow_valid = _resize(flow_valid[crop].float(), size, "bilinear") >= 1 - 1e-6
    if flip:
        image, inverse, valid, flow, flow_valid = [x.flip(-1) for x in (image, inverse, valid, flow, flow_valid)]
        flow[0].neg_()
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    endpoint_x, endpoint_y = xx + flow[0], yy + flow[1]
    endpoint_valid = (endpoint_x >= 0) & (endpoint_x <= size - 1) & (endpoint_y >= 0) & (endpoint_y <= size - 1)
    flow_valid &= torch.isfinite(flow).all(0, keepdim=True) & endpoint_valid[None]
    flow = torch.nan_to_num(flow)
    return image, inverse, valid, flow, flow_valid

class TartanAirPairDataset(Dataset):
    def __init__(self, manifest_path, split="train", input_size=384, augment=False, seed=42, max_depth=80.0):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.root = Path(manifest["root"]).resolve()
        if split not in manifest["splits"]:
            raise ValueError(f"Unknown split {split}")
        self.pairs = manifest["splits"][split]
        self.input_size, self.augment, self.seed, self.epoch, self.max_depth = input_size, augment, seed, 0, max_depth
        trajectories = {k: {x["trajectory"] for x in v} for k, v in manifest["splits"].items()}
        keys = list(trajectories)
        for i, key in enumerate(keys):
            for other in keys[i + 1:]:
                if trajectories[key] & trajectories[other]:
                    raise ValueError(f"Trajectory leakage between {key} and {other}")
        if input_size <= 1:
            raise ValueError("input_size must exceed 1")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.pairs)

    def _path(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"Manifest path escapes root: {relative}")
        return p

    def __getitem__(self, index):
        pair = self.pairs[index]
        try:
            images = [np.asarray(Image.open(self._path(pair[k])).convert("RGB"), dtype=np.float32).copy() / 255 for k in ("rgb0", "rgb1")]
            depths = [np.load(self._path(pair[k]), allow_pickle=False).astype(np.float32) for k in ("depth0", "depth1")]
            flow = np.load(self._path(pair["flow"]), allow_pickle=False).astype(np.float32)
            mask = np.load(self._path(pair["flow_mask"]), allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {pair['trajectory']} frame {pair['frame_id']}: {e}") from e
        h, w = depths[0].shape
        if any(d.shape != (h, w) for d in depths) or any(i.shape != (h, w, 3) for i in images) or flow.shape != (h, w, 2) or mask.shape not in ((h, w), (h, w, 1)):
            raise ValueError(f"Mismatched shapes in {pair['trajectory']} frame {pair['frame_id']}")
        image = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        depth = torch.from_numpy(np.stack(depths))[:, None]
        flow = torch.from_numpy(flow).permute(2, 0, 1)
        # Official TartanAir sample visualization masks all values > 0 black.
        flow_valid = torch.from_numpy(mask.reshape(h, w) == 0)[None]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        flip = self.augment and rng.random() < 0.5
        image, target, valid, flow, flow_valid = transform_pair(image, depth, flow, flow_valid, self.input_size, flip, self.max_depth)
        if self.augment:
            brightness = rng.uniform(0.9, 1.1)
            image = (image * brightness).clamp(0, 1)
        return {"image": image.contiguous(), "target": target.contiguous(), "mask": valid.contiguous(), "flow": flow.contiguous(), "flow_mask": flow_valid.contiguous(), "trajectory": pair["trajectory"], "frame_id": pair["frame_id"]}
