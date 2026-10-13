"""Depth-aware bokeh (shallow depth of field) from ZipDepth relative inverse depth.

The renderer is the classical, scattering-based half of BokehMe (Peng et al.,
CVPR 2022, https://github.com/JuewenPeng/BokehMe, classical_renderer/scatter.py):
every pixel spreads its gamma-linearised colour over a disc whose radius is its
defocus ``K * (d - d_focus)``. Weights ``1 / (r^2 + 0.2)`` let sharp pixels
dominate the blurred pixels they overlap, and a tanh edge anti-aliases the disc.
BokehMe's neural renderer, which repairs errors at occlusion boundaries, is not
included.

``downsample > 1`` is the mobile variant used by the Android app: scatter at
reduced resolution, upsample, and keep the full-resolution input wherever the
defocus radius is below a pixel. ``android/.../BokehRenderer.kt`` mirrors this
module and is checked against ``tests/fixtures/bokeh.json``.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

SOFT_EDGE = 4.0
WEIGHT_EPS = 0.2
FOCUS_WINDOW_RADIUS = 4


def normalize_disparity(inverse_depth: np.ndarray, bounds: Optional[Sequence[float]] = None,
                        percentiles: Tuple[float, float] = (2.0, 98.0)
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Map relative inverse depth to disparity in [0, 1] (1 = nearest).

    Returns the disparity and the (low, high) inverse-depth bounds that were used.
    """
    inv = np.asarray(inverse_depth, np.float32)
    if bounds is None:
        finite = inv[np.isfinite(inv)]
        if not finite.size:
            raise ValueError('Inverse depth has no finite values')
        bounds = np.percentile(finite, percentiles)
    lo, hi = float(bounds[0]), float(bounds[1])
    disparity = np.clip((inv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.nan_to_num(disparity).astype(np.float32), np.array([lo, hi])


def focus_disparity(disparity: np.ndarray, point: Tuple[float, float],
                    radius: int = FOCUS_WINDOW_RADIUS) -> float:
    """Median disparity in a small window around ``point`` = (x, y) in [0, 1]."""
    h, w = disparity.shape
    x = int(round(min(max(point[0], 0.0), 1.0) * (w - 1)))
    y = int(round(min(max(point[1], 0.0), 1.0) * (h - 1)))
    window = disparity[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
    return float(np.median(window))


def defocus_map(disparity: np.ndarray, focus: float, strength: float) -> np.ndarray:
    """Signed blur radius in pixels: positive in front of the focal plane, negative behind.

    ``strength`` is the radius for a full disparity range, as a fraction of the longer side.
    """
    return (strength * max(disparity.shape) * (disparity - focus)).astype(np.float32)


def scatter(image: np.ndarray, defocus: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """BokehMe classical renderer: scatter HxWx3 linear colour by signed defocus HxW."""
    src = torch.from_numpy(np.ascontiguousarray(np.asarray(image, np.float32).transpose(2, 0, 1)))
    radius = torch.from_numpy(np.abs(np.asarray(defocus, np.float32))).to(device)
    h, w = radius.shape
    src = torch.cat([torch.ones(1, h, w), src]).to(device)  # channel 0 accumulates weights
    reach = torch.floor(radius).to(torch.int32) + 1
    max_reach = int(reach.max())
    norm = 1.0 / (radius * radius + WEIGHT_EPS)
    acc = torch.zeros(4, h + 2 * max_reach, w + 2 * max_reach, device=device)
    masks = [(reach >= ring).float() for ring in range(max_reach + 1)]
    for dy in range(-max_reach, max_reach + 1):
        for dx in range(-max_reach, max_reach + 1):
            # The CUDA kernel visits a square of half-size floor(r) + 1 around each source pixel.
            mask = masks[max(abs(dy), abs(dx))]
            weight = (0.5 + 0.5 * torch.tanh(SOFT_EDGE * (radius - math.hypot(dy, dx)))) * norm * mask
            y0, x0 = max_reach + dy, max_reach + dx
            acc[:, y0:y0 + h, x0:x0 + w] += weight * src
    acc = acc[:, max_reach:max_reach + h, max_reach:max_reach + w]
    return (acc[1:] / acc[0]).permute(1, 2, 0).cpu().numpy()


def block_mean(x: np.ndarray, factor: int) -> np.ndarray:
    h, w = x.shape[:2]
    if h % factor or w % factor:
        raise ValueError(f'{h}x{w} is not divisible by downsample factor {factor}')
    return x.reshape(h // factor, factor, w // factor, factor, *x.shape[2:]).mean(axis=(1, 3))


def upsample_bilinear(x: np.ndarray, factor: int) -> np.ndarray:
    """Bilinear upsampling with half-pixel centres and edge clamping (cv2.INTER_LINEAR)."""
    def axis(n):
        src = np.clip((np.arange(n * factor) + 0.5) / factor - 0.5, 0, n - 1)
        low = np.floor(src).astype(np.int64)
        return low, np.minimum(low + 1, n - 1), (src - low).astype(np.float32)

    y0, y1, ty = axis(x.shape[0])
    x0, x1, tx = axis(x.shape[1])
    extra = (1,) * (x.ndim - 2)
    ty = ty.reshape(-1, 1, *extra)
    tx = tx.reshape(1, -1, *extra)
    top = x[y0][:, x0] * (1 - tx) + x[y0][:, x1] * tx
    bottom = x[y1][:, x0] * (1 - tx) + x[y1][:, x1] * tx
    return (top * (1 - ty) + bottom * ty).astype(np.float32)


def render_bokeh(image: np.ndarray, disparity: np.ndarray, focus: float,
                 strength: float = 0.03, gamma: float = 4.0, downsample: int = 1,
                 device: str = 'cpu') -> np.ndarray:
    """Render bokeh for an HxWx3 RGB image in [0, 1] focused at disparity ``focus``."""
    image = np.clip(np.asarray(image, np.float32), 0.0, 1.0)
    disparity = np.asarray(disparity, np.float32)
    if image.shape[:2] != disparity.shape:
        raise ValueError(f'Image {image.shape[:2]} and disparity {disparity.shape} differ')
    defocus = defocus_map(disparity, focus, strength)
    linear = image ** gamma
    if downsample == 1:
        return np.clip(scatter(linear, defocus, device) ** (1.0 / gamma), 0.0, 1.0)
    small = scatter(block_mean(linear, downsample), block_mean(defocus, downsample) / downsample, device)
    blurred = upsample_bilinear(np.clip(small, 0.0, 1.0) ** (1.0 / gamma), downsample)
    alpha = np.clip(np.abs(defocus) - 0.5, 0.0, 1.0)[..., None]
    return np.clip(image * (1 - alpha) + blurred * alpha, 0.0, 1.0).astype(np.float32)


class FocusTracker:
    """Per-video state: smoothed disparity range and focus, so the bokeh does not pump.

    ``focus_ema`` blends the focus disparity towards the value under the focus
    point; a scene cut or a new focus point snaps it instead.
    """

    def __init__(self, range_ema: float = 0.1, focus_ema: float = 0.3):
        self.range_ema, self.focus_ema = range_ema, focus_ema
        self.reset()

    def reset(self):
        self.bounds: Optional[np.ndarray] = None
        self.focus: Optional[float] = None

    def update(self, inverse_depth: np.ndarray, point: Tuple[float, float],
               snap: bool = False) -> Tuple[np.ndarray, float]:
        _, current = normalize_disparity(inverse_depth)
        if snap or self.bounds is None:
            self.bounds = current
        else:
            self.bounds = self.bounds * (1 - self.range_ema) + current * self.range_ema
        disparity, _ = normalize_disparity(inverse_depth, self.bounds)
        target = focus_disparity(disparity, point)
        if snap or self.focus is None:
            self.focus = target
        else:
            self.focus = self.focus * (1 - self.focus_ema) + target * self.focus_ema
        return disparity, self.focus
