#!/usr/bin/env python
# encoding: utf-8
"""TartanAir 视频序列、同步数据增强和不补齐样本的分布式采样器。"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

TAG_FLOAT = 202021.25


def _resize(array, size, mode="bilinear"):
    """Resize HWC float32 data while preserving singleton/empty channels."""
    if array.shape[-1] == 0:
        return np.empty((*size, 0), dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    resized = F.interpolate(tensor, size=size, mode=mode, **kwargs)
    return resized[0].permute(1, 2, 0).numpy().copy()


class RandomScale:
    """同步缩放；最后 last_ch 个通道为交替 u/v 光流，逆深度不缩放数值。"""
    def __init__(self, scale_limit, last_ch, mask_channels=None):
        if len(scale_limit) != 2 or not 0 < scale_limit[0] <= scale_limit[1]:
            raise ValueError("scale_limit must be a positive (min, max) pair")
        if last_ch < 0 or last_ch % 2:
            raise ValueError("last_ch must contain complete two-channel flows")
        self.scale_limit, self.last_ch = scale_limit, last_ch
        self.mask_channels = mask_channels

    def __call__(self, x):
        if x.ndim != 3 or self.last_ch > x.shape[-1]:
            raise ValueError("Expected HWC input with enough flow channels")
        scale = np.random.uniform(*self.scale_limit)
        height, width = x.shape[:2]
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        resized = _resize(x, size)
        if self.mask_channels is not None:
            # A vector is valid only if all interpolation support was valid.
            resized[..., self.mask_channels] = (_resize(
                x[..., self.mask_channels], size) >= 1 - 1e-6).astype(np.float32)
        if self.last_ch:
            resized[..., -self.last_ch::2] *= size[1] / width
            resized[..., -self.last_ch + 1::2] *= size[0] / height
        return resized


class RandomHorizontalFlip:
    """以 0.5 的概率同步水平翻转，只改变光流水平分量的符号。"""
    def __init__(self, last_ch):
        if last_ch < 0 or last_ch % 2:
            raise ValueError("last_ch must contain complete two-channel flows")
        self.last_ch = last_ch

    def __call__(self, x):
        if x.ndim != 3 or self.last_ch > x.shape[-1]:
            raise ValueError("Expected HWC input with enough flow channels")
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            if self.last_ch:
                x[..., -self.last_ch::2] *= -1
        return x


class _Crop:
    def __init__(self, size, random):
        self.size, self.random = size, random

    def __call__(self, *, image):
        height, width = image.shape[:2]
        if min(height, width) < self.size:
            raise ValueError(f"Image size {(height, width)} is smaller than crop_size={self.size}; reduce crop_size")
        if self.random:
            top = np.random.randint(height - self.size + 1)
            left = np.random.randint(width - self.size + 1)
        else:
            top, left = (height - self.size) // 2, (width - self.size) // 2
        return {"image": image[top:top + self.size, left:left + self.size].copy()}


class DepthVideoDataset(Dataset):
    """读取 RGB、逆深度、双向像素光流和有效掩码。

    根目录或轨迹中递归发现 image_left，不限制 TartanAir 的大小写/层数。
    光流位于同级 <数据根>_flow，亦支持轨迹内部 f_flow/b_flow/f_mask/b_mask。
    四类文件均用帧对第一帧的图像 stem 命名，b_flow 表示第二帧到第一帧。
    磁盘 mask 的 0 有效、非 0 无效；返回 mask 的 1 有效、0 无效。
    每条轨迹最后一个完整序列用于 val，其余最多 50 个用于 train。
    """
    def __init__(self, mode, data_dirs=None, crop_size=384, seq_len=12):
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'")
        if not isinstance(seq_len, int) or seq_len < 2:
            raise ValueError("seq_len must be an integer of at least 2")
        if not isinstance(crop_size, int) or crop_size <= 0:
            raise ValueError("crop_size must be a positive integer")
        if data_dirs is None:
            data_dirs = [Path(__file__).resolve().parent / "data" / "tartanair"]
        elif isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.mode, self.crop_size, self.seq_len = mode, crop_size, seq_len
        self.data_paths = []
        seen = set()
        for directory in data_dirs:
            root = Path(directory).resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {root}")
            if root.name == "image_left":
                root = root.parent
            image_dirs = ([root / "image_left"] if (root / "image_left").is_dir()
                          else sorted(root.rglob("image_left")))
            for image_dir in image_dirs:
                if not image_dir.is_dir() or image_dir in seen:
                    continue
                seen.add(image_dir)
                trajectory = image_dir.parent
                depth_dir = trajectory / "depth_left"
                if not depth_dir.is_dir():
                    raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
                images = sorted(p for p in image_dir.iterdir()
                                if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
                depths = []
                for image in images:
                    depth = depth_dir / f"{image.stem}_depth.npy"
                    if not depth.is_file():
                        depth = depth_dir / f"{image.stem}.npy"
                    if not depth.is_file():
                        raise FileNotFoundError(f"Missing depth for image: {image}")
                    depths.append(depth)
                flow_trajectory = self._flow_trajectory(root, trajectory)
                sequence_count = len(images) // seq_len
                indices = (range(min(50, max(0, sequence_count - 1))) if mode == "train"
                           else range(max(0, sequence_count - 1), sequence_count))
                for sequence in indices:
                    start = sequence * seq_len
                    frames = images[start:start + seq_len]
                    frame_ids = [p.stem.split("_")[0] for p in frames]
                    if all(frame.isdigit() for frame in frame_ids):
                        if any(int(b) != int(a) + 1 for a, b in zip(frame_ids, frame_ids[1:])):
                            raise ValueError(f"Non-adjacent frames in trajectory: {trajectory}")
                    paths = []
                    for j, image in enumerate(frames):
                        forward = image.stem
                        backward = frames[j - 1].stem if j else None
                        row = [str(image), str(depths[start + j]),
                               str(flow_trajectory / "f_flow" / f"{forward}.npy") if j < seq_len - 1 else None,
                               str(flow_trajectory / "b_flow" / f"{backward}.npy") if j else None,
                               str(flow_trajectory / "f_mask" / f"{forward}.npy") if j < seq_len - 1 else None,
                               str(flow_trajectory / "b_mask" / f"{backward}.npy") if j else None]
                        for path in row[2:]:
                            if path is not None and not Path(path).is_file():
                                raise FileNotFoundError(f"Missing generated flow/mask: {path}. Run scripts/estimate_flow.py first.")
                        paths.append(row)
                    self.data_paths.append(["TartanAir", paths])
        self.scale = {"TartanAir": RandomScale((0.8, 0.85), 4 * (seq_len - 1), slice(4 * seq_len, 6 * seq_len - 2))}
        self.flip = {"TartanAir": RandomHorizontalFlip(4 * (seq_len - 1))}
        self.transform = {"TartanAir": _Crop(crop_size, random=mode == "train")}

    @staticmethod
    def _flow_trajectory(root, trajectory):
        if (trajectory / "f_flow").is_dir():
            return trajectory
        for ancestor in (trajectory, *trajectory.parents):
            if ancestor.name.lower().startswith("tartanair") and not ancestor.name.lower().endswith("_flow"):
                return ancestor.with_name(ancestor.name + "_flow") / trajectory.relative_to(ancestor)
        return root.with_name(root.name + "_flow") / trajectory.relative_to(root)

    @staticmethod
    def _read_flow(flow_path, mask_path, size):
        if flow_path is None:
            return np.empty((*size, 0), np.float32), np.empty((*size, 0), np.float32)
        flow = np.load(flow_path, allow_pickle=False).astype(np.float32)
        raw_mask = np.load(mask_path, allow_pickle=False)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError(f"Flow must have shape [H,W,2]: {flow_path}")
        if raw_mask.shape == (*flow.shape[:2], 1):
            raw_mask = raw_mask[..., 0]
        if raw_mask.shape != flow.shape[:2]:
            raise ValueError(f"Mask and flow shapes differ: {mask_path}")
        valid = (raw_mask == 0) & np.isfinite(flow).all(axis=-1)
        flow = np.where(valid[..., None], flow, 0)
        height, width = flow.shape[:2]
        flow = _resize(flow, size)
        flow[..., 0] *= size[1] / width
        flow[..., 1] *= size[0] / height
        mask = (_resize(valid[..., None].astype(np.float32), size) >= 1 - 1e-6).astype(np.float32)
        return flow, mask

    def __getitem__(self, item):
        label, set_paths = self.data_paths[item]
        images, depths, f_flows, b_flows, f_masks, b_masks = [], [], [], [], [], []
        size = None
        for image_path, depth_path, fp, bp, fm, bm in set_paths:
            with Image.open(image_path) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.float32) / 127.5 - 1
            if size is None:
                size = image.shape[:2]
            if image.shape[:2] != size:
                raise ValueError(f"Image sizes differ in sequence: {image_path}")
            metric_depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
            if metric_depth.shape == (*size, 1):
                metric_depth = metric_depth[..., 0]
            if metric_depth.shape != size:
                raise ValueError(f"Depth and image shapes differ: {depth_path}")
            valid = np.isfinite(metric_depth) & (metric_depth > 0)
            depth = np.divide(1, metric_depth, out=np.zeros_like(metric_depth), where=valid)[..., None]
            f_flow, f_mask = self._read_flow(fp, fm, size)
            b_flow, b_mask = self._read_flow(bp, bm, size)
            for target, value in zip((images, depths, f_flows, b_flows, f_masks, b_masks),
                                     (image, depth, f_flow, b_flow, f_mask, b_mask)):
                target.append(value)
        packed = np.concatenate([np.concatenate(values, axis=-1)
                                 for values in (images, depths, f_masks, b_masks, f_flows, b_flows)], axis=-1)
        if self.mode == "train":
            packed = self.scale[label](packed)
            packed = self.flip[label](packed)
        packed = self.transform[label](image=packed)["image"]
        t = self.seq_len
        channels = (3 * t, t, t - 1, t - 1, 2 * (t - 1), 2 * (t - 1))
        arrays = np.split(packed, np.cumsum(channels)[:-1], axis=-1)
        sample = {}
        for name, array, frames, count in zip(
                ("image", "depth", "f_masks", "b_masks", "f_flows", "b_flows"),
                arrays, (t, t, t - 1, t - 1, t - 1, t - 1), (3, 1, 1, 1, 2, 2)):
            array = array.reshape(self.crop_size, self.crop_size, frames, count)
            sample[name] = torch.from_numpy(array.copy()).permute(2, 3, 0, 1).contiguous()
        yy, xx = torch.meshgrid(torch.arange(self.crop_size), torch.arange(self.crop_size), indexing="ij")
        for flow_key, mask_key in (("f_flows", "f_masks"), ("b_flows", "b_masks")):
            flow = sample[flow_key]
            x_end, y_end = xx + flow[:, 0], yy + flow[:, 1]
            in_bounds = ((x_end >= 0) & (x_end <= self.crop_size - 1)
                         & (y_end >= 0) & (y_end <= self.crop_size - 1))
            sample[mask_key] *= in_bounds[:, None]
        return sample

    def __len__(self):
        return len(self.data_paths)


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """分配全部样本，不补齐、不重复；各 rank 数量最多相差 1。"""
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None or rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Initialize torch.distributed or supply num_replicas and rank")
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer")
        if not isinstance(rank, int) or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.epoch, self.shuffle = 0, shuffle
        count, rest = divmod(len(dataset), num_replicas)
        self.num_samples = count + int(rank < rest)
        self.total_size = len(dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.epoch)
        indices = (torch.randperm(len(self.dataset), generator=generator).tolist()
                   if self.shuffle else list(range(len(self.dataset))))
        indices = indices[self.rank::self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "tartanair")
    parser.add_argument("--mode", choices=("train", "val"), default="val")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=384)
    args = parser.parse_args()
    dataset = DepthVideoDataset(args.mode, args.data_dir, args.crop_size, args.seq_len)
    print(f"Loaded {len(dataset)} {args.mode} sequences")
    for sample in DataLoader(dataset, batch_size=1, num_workers=0):
        print({key: tuple(value.shape) for key, value in sample.items()})
        break
