"""Masked affine-invariant supervision and flow-aligned temporal error loss.

Targets are inverse depth, never quantized PNG units. The temporal term compares
prediction *errors*, so genuine depth changes caused by camera motion are not
penalized. Both frames share one affine normalization for this term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_normalize(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Median/mean absolute deviation normalization over all but batch axis.

    Empty maps produce differentiable zeros. Invalid values are replaced before
    reductions, because multiplying NaN by a zero mask does not remove NaN.
    """
    value = value.float()
    valid = mask.bool() & torch.isfinite(value)
    clean = torch.where(valid, value, torch.zeros_like(value))
    flat = clean.flatten(1)
    flat_mask = valid.flatten(1)
    nonempty = flat_mask.any(1)
    # Stable sorting fixes the selected pixel when BF16 creates tied median
    # values, avoiding device-dependent median tie selection during backward.
    median_input = torch.where(flat_mask, flat, torch.full_like(flat, float('inf')))
    ordered = median_input.sort(dim=1, stable=True).values
    middle = ((flat_mask.sum(1) - 1).clamp_min(0) // 2).unsqueeze(1)
    median = ordered.gather(1, middle).squeeze(1)
    median = torch.where(nonempty, median, torch.zeros_like(median))
    shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    center = median.reshape(shape)
    count = valid.flatten(1).sum(1).clamp_min(1).reshape(shape)
    scale = ((clean - center).abs() * valid).flatten(1).sum(1).reshape(shape) / count
    normalized = (clean - center) / scale.clamp_min(eps)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    return torch.where(valid, value, torch.zeros_like(value)).sum() / valid.sum().clamp_min(1)


def flow_warp_next(value: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample frame t+1 at x + forward_flow(t->t+1), returning in-bounds mask.

    value: [B,C,H,W]; flow: [B,2,H,W] in resized image pixel units.
    """
    batch, _, height, width = value.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=value.device, dtype=torch.float32),
        torch.arange(width, device=value.device, dtype=torch.float32), indexing='ij',
    )
    finite = torch.isfinite(flow).all(1, keepdim=True)
    flow = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
    sx, sy = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
    inside = ((sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1))[:, None] & finite
    # grid_sample CUDA backward uses atomic scatter with no deterministic path.
    # Explicit bilinear gather is equivalent and torch's deterministic mode
    # supplies a deterministic gather backward implementation.
    x0, y0 = sx.floor().long(), sy.floor().long()
    wx, wy = sx - x0, sy - y0
    flat = value.float().flatten(2)
    warped = torch.zeros_like(value, dtype=torch.float32)
    for ix, iy, weight in ((x0, y0, (1 - wx) * (1 - wy)),
                           (x0 + 1, y0, wx * (1 - wy)),
                           (x0, y0 + 1, (1 - wx) * wy),
                           (x0 + 1, y0 + 1, wx * wy)):
        neighbor_valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        address = (iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1)).flatten(1)
        sampled = flat.gather(2, address[:, None].expand(-1, value.shape[1], -1)).reshape_as(value)
        warped = warped + sampled * (weight * neighbor_valid)[:, None]
    return warped, inside


class TemporalDepthLoss(nn.Module):
    """Spatial SSI + multiscale gradients + weighted temporal prediction error."""

    def __init__(self, temporal_weight: float = 0.1, alpha_ssi: float = 1.0,
                 alpha_grad: float = 2.0, grad_scales: int = 4):
        super().__init__()
        if temporal_weight < 0 or grad_scales < 1:
            raise ValueError('temporal_weight must be nonnegative; grad_scales must be positive')
        self.temporal_weight = temporal_weight
        self.alpha_ssi = alpha_ssi
        self.alpha_grad = alpha_grad
        self.grad_scales = grad_scales

    def forward(self, pred, target, mask, flow, flow_mask):
        if pred.ndim != 5 or pred.shape[1:3] != (2, 1) or pred.shape != target.shape:
            raise ValueError('pred and target must have equal shape [B,2,1,H,W]')
        pred, target = pred.float(), target.float()
        valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(target) & (target > 0)
        flat_pred = pred.flatten(0, 1)
        flat_target = target.flatten(0, 1)
        flat_valid = valid.flatten(0, 1)
        p = masked_normalize(flat_pred, flat_valid)
        t = masked_normalize(flat_target, flat_valid)
        ssi = masked_mean((p - t).abs(), flat_valid)
        gradient = p.sum() * 0.0
        scales_used = 0
        for index in range(self.grad_scales):
            scale = 2 ** index
            if scale > min(p.shape[-2:]):
                break
            ps, ts, ms = p, t, flat_valid
            if scale > 1:
                ps, ts = F.avg_pool2d(p, scale), F.avg_pool2d(t, scale)
                ms = F.avg_pool2d(flat_valid.float(), scale) >= 1.0 - 1e-6
            dx = (ps[..., 1:] - ps[..., :-1]) - (ts[..., 1:] - ts[..., :-1])
            dy = (ps[..., 1:, :] - ps[..., :-1, :]) - (ts[..., 1:, :] - ts[..., :-1, :])
            mx = ms[..., 1:] & ms[..., :-1]
            my = ms[..., 1:, :] & ms[..., :-1, :]
            numerator = torch.where(mx, dx.abs(), torch.zeros_like(dx)).sum()
            numerator = numerator + torch.where(my, dy.abs(), torch.zeros_like(dy)).sum()
            gradient = gradient + numerator / (mx.sum() + my.sum()).clamp_min(1)
            scales_used += 1
        gradient = gradient / max(scales_used, 1)

        # First dimension remains B; both frames are reduced together.
        pair_p = masked_normalize(pred, valid)
        pair_t = masked_normalize(target, valid)
        error = pair_p - pair_t
        error_next, inside = flow_warp_next(error[:, 1], flow)
        valid_next, _ = flow_warp_next(valid[:, 1].float(), flow)
        temporal_mask = valid[:, 0] & inside & flow_mask.bool() & (valid_next >= 1.0 - 1e-6)
        temporal = masked_mean((error[:, 0] - error_next).abs(), temporal_mask)
        spatial = self.alpha_ssi * ssi + self.alpha_grad * gradient
        total = spatial + self.temporal_weight * temporal
        return total, {
            'ssi': ssi.detach(), 'gradient': gradient.detach(), 'spatial': spatial.detach(),
            'temporal': temporal.detach(), 'total': total.detach(),
            'valid_temporal_pixels': temporal_mask.sum().detach(),
        }
