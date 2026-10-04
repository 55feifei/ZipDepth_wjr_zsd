"""Shared mobile preprocessing, causal smoothing and honest latency summaries."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from zipdepth.model.architecture import create_model
from zipdepth.utils.model_utils import strip_state_dict_prefixes


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def preprocess_bgr(frame: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected HWC BGR image')
    h, w = frame.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    crop = frame[y:y+side, x:x+side]
    rgb = cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), rgb


def load_model(checkpoint: str | Path, device='cpu', fused=False):
    model = create_model(variant='base', global_mode='balanced', upsample_unfold=False)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(strip_state_dict_prefixes(state), strict=True)
    model.eval()
    if fused:
        from zipdepth.utils.model_utils import fuse_remaining_conv_bn
        model.fuse_for_inference()
        fuse_remaining_conv_bn(model)
    return model.to(device)


class GatedEMA:
    def __init__(self, history_weight=.3, rgb_tau=.08, edge_threshold=.1, scene_threshold=.25):
        self.history_weight = history_weight
        self.rgb_tau = rgb_tau
        self.edge_threshold = edge_threshold
        self.scene_threshold = scene_threshold
        self.reset()

    def reset(self):
        self.previous_rgb = self.previous_depth = None
        self.scene_reset = True

    def update(self, rgb, depth):
        rgb, depth = np.asarray(rgb, np.float32), np.asarray(depth, np.float32)
        if rgb.shape != (*depth.shape, 3) or depth.ndim != 2:
            raise ValueError('RGB/depth shapes must be HWC/HW and agree')
        if not np.isfinite(rgb).all() or not np.isfinite(depth).all():
            raise ValueError('Non-finite smoothing input')
        if np.any(rgb < 0) or np.any(rgb > 1):
            raise ValueError('RGB must be in [0,1]')
        first = self.previous_depth is None or self.previous_depth.shape != depth.shape
        difference = None if first else np.abs(rgb - self.previous_rgb).mean(axis=2)
        self.scene_reset = first or float(difference.mean()) > self.scene_threshold
        if self.scene_reset:
            output = depth.copy()
        else:
            weight = self.history_weight * np.exp(-difference / self.rgb_tau)
            edge = np.zeros_like(depth)
            edge[:, 1:] += np.abs(depth[:, 1:] - depth[:, :-1])
            edge[1:, :] += np.abs(depth[1:, :] - depth[:-1, :])
            lo, hi = np.percentile(depth, [2, 98])
            edge /= max(float(hi - lo), 1e-6)
            weight[edge > self.edge_threshold] = 0
            output = depth * (1 - weight) + self.previous_depth * weight
        self.previous_rgb, self.previous_depth = rgb.copy(), output.copy()
        return output


class StableColorizer:
    def __init__(self, fixed_range=None, update_rate=.05):
        self.fixed_range = fixed_range
        self.update_rate = update_rate
        self.bounds = None

    def reset(self):
        self.bounds = None

    def update(self, depth, reset=False):
        if reset:
            self.reset()
        if self.fixed_range is not None:
            lo, hi = self.fixed_range
        else:
            current = np.percentile(depth, [2, 98])
            self.bounds = current if self.bounds is None else self.bounds * (1-self.update_rate) + current * self.update_rate
            lo, hi = self.bounds
        gray = np.clip((depth-lo) / max(float(hi-lo), 1e-6) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


class DepthRuntime:
    def __init__(self, model_path, size=256, device='cpu'):
        self.size, self.device = size, device
        self.onnx = str(model_path).endswith('.onnx')
        if self.onnx:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(model_path), opts, providers=['CPUExecutionProvider'])
            expected = self.session.get_inputs()[0].shape
            if expected != [1, 3, size, size]:
                raise ValueError(f'ONNX input {expected} incompatible with size={size}')
        else:
            self.model = load_model(model_path, device, fused=True)

    @torch.inference_mode()
    def predict(self, frame):
        start = time.perf_counter()
        tensor, rgb = preprocess_bgr(frame, self.size)
        if self.onnx:
            pre = time.perf_counter()
            depth = self.session.run(['depth'], {'image': tensor})[0][0, 0]
        else:
            x = torch.from_numpy(tensor).to(self.device)
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            pre = time.perf_counter()
            depth = self.model(x)[0, 0].float().cpu().numpy()
        done = time.perf_counter()
        if not np.isfinite(depth).all():
            raise ValueError('Model returned non-finite depth')
        return depth, rgb, {'preprocess_ms': (pre-start)*1000, 'inference_ms': (done-pre)*1000}


def timing_summary(values):
    a = np.asarray(values, np.float64)
    if not a.size:
        return {'count': 0}
    return {'count': len(a), 'mean_ms': float(a.mean()), 'p50_ms': float(np.percentile(a, 50)),
            'p95_ms': float(np.percentile(a, 95)), 'fps': float(1000 / max(a.mean(), 1e-9))}
