"""Back-project ZipDepth predictions into coloured 3D point clouds.

ZipDepth predicts affine-invariant (relative) inverse depth, so the geometry is
only known up to an unknown scale and shift. The clouds built here place the
scene inside a virtual [near, far] range: their shape is meaningful for
visualisation, their absolute size is not metric.
"""

from typing import Optional, Tuple

import numpy as np

Intrinsics = Tuple[float, float, float, float]

# OpenCV camera frame (x right, y down, z forward) -> OpenGL / Open3D viewer
# frame (x right, y up, z backward). PLY viewers also expect the latter.
OPENCV_TO_OPENGL = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> Intrinsics:
    """Pinhole intrinsics (fx, fy, cx, cy) from a horizontal field of view."""
    if not 0 < fov_deg < 180:
        raise ValueError(f'fov_deg must be in (0, 180), got {fov_deg}')
    fx = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2)
    return float(fx), float(fx), (width - 1) / 2.0, (height - 1) / 2.0


def scale_intrinsics(intrinsics: Intrinsics, sx: float, sy: float) -> Intrinsics:
    """Rescale intrinsics given for one resolution to a resized image."""
    fx, fy, cx, cy = intrinsics
    return fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5


class InverseDepthMapper:
    """Turn relative inverse depth into positive depth inside [near, far].

    The affine ambiguity is fixed by mapping the low/high inverse-depth
    percentiles to 1/far and 1/near; values outside are clamped to the range.
    For video, ``ema`` (0..1) smooths the percentile bounds so the scene does
    not breathe from frame to frame; ``None`` uses each frame on its own.
    """

    def __init__(self, near: float = 1.0, far: float = 10.0,
                 percentiles: Tuple[float, float] = (2.0, 98.0),
                 ema: Optional[float] = None):
        if not 0 < near < far:
            raise ValueError(f'Need 0 < near < far, got near={near}, far={far}')
        if not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError(f'Invalid percentiles {percentiles}')
        if ema is not None and not 0 < ema <= 1:
            raise ValueError(f'ema must be in (0, 1], got {ema}')
        self.near, self.far = float(near), float(far)
        self.percentiles = percentiles
        self.ema = ema
        self.bounds: Optional[np.ndarray] = None

    def reset(self):
        self.bounds = None

    def __call__(self, inverse_depth: np.ndarray) -> np.ndarray:
        inv = np.asarray(inverse_depth, dtype=np.float32)
        if inv.ndim != 2:
            raise ValueError(f'Expected HxW inverse depth, got shape {inv.shape}')
        finite = np.isfinite(inv)
        if not finite.any():
            raise ValueError('Inverse depth has no finite values')
        current = np.percentile(inv[finite], self.percentiles).astype(np.float64)
        if self.ema is None or self.bounds is None:
            self.bounds = current
        else:
            self.bounds = self.bounds * (1 - self.ema) + current * self.ema
        lo, hi = self.bounds
        normalized = np.clip((inv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        disparity = 1.0 / self.far + normalized * (1.0 / self.near - 1.0 / self.far)
        depth = (1.0 / disparity).astype(np.float32)
        depth[~finite] = np.nan
        return depth


def depth_to_points(depth: np.ndarray, intrinsics: Intrinsics,
                    colors: Optional[np.ndarray] = None, stride: int = 1,
                    edge_threshold: float = 0.0
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Back-project a depth map with a pinhole model.

    Args:
        depth: HxW positive depth (NaN/<=0 pixels are dropped).
        intrinsics: (fx, fy, cx, cy) in pixels of ``depth``.
        colors: optional HxWx3 RGB, uint8 or float in [0, 1].
        stride: keep every ``stride``-th pixel along each axis.
        edge_threshold: drop "flying" points whose depth differs from a
            sampled neighbour by more than this fraction of their own depth
            (0 disables the filter).

    Returns:
        points Nx3 float32 in the OpenCV camera frame, colours Nx3 float32 in
        [0, 1] (or None).
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f'Expected HxW depth, got shape {depth.shape}')
    if stride < 1:
        raise ValueError(f'stride must be >= 1, got {stride}')
    if colors is not None and colors.shape[:2] != depth.shape:
        raise ValueError(f'Colour shape {colors.shape[:2]} != depth shape {depth.shape}')

    fx, fy, cx, cy = intrinsics
    z = depth[::stride, ::stride]
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride].astype(np.float32)

    with np.errstate(invalid='ignore'):
        valid = np.isfinite(z) & (z > 0)
    if edge_threshold > 0:
        jump = np.zeros_like(z)
        dx = np.abs(np.diff(z, axis=1))
        dy = np.abs(np.diff(z, axis=0))
        jump[:, 1:] = np.fmax(jump[:, 1:], dx)
        jump[:, :-1] = np.fmax(jump[:, :-1], dx)
        jump[1:, :] = np.fmax(jump[1:, :], dy)
        jump[:-1, :] = np.fmax(jump[:-1, :], dy)
        with np.errstate(invalid='ignore', divide='ignore'):
            valid &= ~(jump / z > edge_threshold)

    z = z[valid]
    points = np.stack([(u[valid] - cx) * z / fx, (v[valid] - cy) * z / fy, z], axis=1)

    point_colors = None
    if colors is not None:
        c = colors[::stride, ::stride][valid]
        point_colors = (c.astype(np.float32) / 255.0 if c.dtype == np.uint8
                        else np.clip(c.astype(np.float32), 0.0, 1.0))
    return points.astype(np.float32), point_colors
