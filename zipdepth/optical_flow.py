"""GMFlow inference using the local, unmodified GMFlow checkout.

Pair inputs are RGB in [0, 255]. Flows use original-image pixel units;
the on-disk masks use 1 for occluded, inconsistent or out-of-frame pixels.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMFLOW_ROOT = PROJECT_ROOT / "gmflow" / "gmflow"
DEFAULT_CHECKPOINT = DEFAULT_GMFLOW_ROOT / "pretrained" / "gmflow_things-e9887eda.pth"


def _load_gmflow(root: Path):
    """Use a private package name so another installed gmflow cannot win."""
    package = root / "gmflow"
    if not (package / "gmflow.py").is_file():
        raise FileNotFoundError(f"GMFlow model not found under {package}")
    name = "_zipdepth_local_gmflow"
    if name in sys.modules:
        loaded = Path(sys.modules[name].__file__).resolve().parent
        if loaded != package.resolve():
            raise RuntimeError(f"A different GMFlow checkout is already loaded: {loaded}")
    else:
        spec = importlib.util.spec_from_file_location(
            name, package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    model = importlib.import_module(f"{name}.gmflow").GMFlow
    consistency = importlib.import_module(
        f"{name}.geometry"
    ).forward_backward_consistency_check
    return model, consistency


def _rgb_tensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("NumPy RGB inputs must have shape [H, W, 3]")
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        tensor = image.detach()
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("Tensor RGB inputs must have shape [3, H, W]")
    else:
        raise TypeError("RGB input must be a NumPy array or torch.Tensor")
    tensor = tensor.to(dtype=torch.float32)
    if min(tensor.shape[-2:]) < 2:
        raise ValueError("Both image dimensions must be at least 2")
    if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 255:
        raise ValueError("Pair inputs must contain finite RGB values in [0, 255]")
    return tensor.unsqueeze(0)


class GMFlowEstimator:
    """Pretrained single-scale GMFlow; one model reused across all frame pairs."""

    def __init__(
        self,
        checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
        gmflow_root: Union[str, Path] = DEFAULT_GMFLOW_ROOT,
        device: str = "auto",
        inference_size: Optional[Sequence[int]] = None,
    ):
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"GMFlow checkpoint does not exist: {checkpoint}")
        if inference_size is not None:
            if len(inference_size) != 2 or any(int(v) < 2 for v in inference_size):
                raise ValueError("inference_size must be (height, width), each >= 2")
            inference_size = tuple(int(v) for v in inference_size)
        self.inference_size = inference_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but this Python/PyTorch environment has no CUDA")
        model_class, self._consistency_check = _load_gmflow(Path(gmflow_root).resolve())
        self.model = model_class().to(self.device).eval()
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = checkpoint_data.get("model", checkpoint_data)
        self.model.load_state_dict(weights, strict=True)

    @torch.inference_mode()
    def infer_pair(self, image0, image1):
        """Return f_flow/b_flow [H,W,2] float32 and invalid masks [H,W] uint8.

        Forward maps image0 -> image1. Backward maps image1 -> image0;
        backward flow is estimated by the model, never by negating forward flow.
        """
        image0, image1 = _rgb_tensor(image0), _rgb_tensor(image1)
        if image0.shape != image1.shape:
            raise ValueError("The two frames must have identical image dimensions")
        height, width = image0.shape[-2:]
        image0, image1 = image0.to(self.device), image1.to(self.device)
        if self.inference_size is not None:
            image0 = F.interpolate(image0, self.inference_size, mode="bilinear", align_corners=True)
            image1 = F.interpolate(image1, self.inference_size, mode="bilinear", align_corners=True)
        infer_h, infer_w = image0.shape[-2:]
        # Single-scale features are /8; the Swin split of 2 needs multiples of 16.
        pad_h = math.ceil(infer_h / 16) * 16 - infer_h
        pad_w = math.ceil(infer_w / 16) * 16 - infer_w
        image0 = F.pad(image0, (0, pad_w, 0, pad_h), mode="replicate")
        image1 = F.pad(image1, (0, pad_w, 0, pad_h), mode="replicate")
        result = self.model(
            image0, image1, attn_splits_list=[2], corr_radius_list=[-1],
            prop_radius_list=[-1], pred_bidir_flow=True,
        )
        flows = result["flow_preds"][-1][..., :infer_h, :infer_w]
        if (infer_h, infer_w) != (height, width):
            flows = F.interpolate(flows, (height, width), mode="bilinear", align_corners=True)
            flows[:, 0] *= width / infer_w
            flows[:, 1] *= height / infer_h
        if not torch.isfinite(flows).all():
            raise RuntimeError("GMFlow produced nonfinite optical flow")
        forward, backward = flows[:1], flows[1:2]
        f_occ, b_occ = self._consistency_check(forward, backward)
        y, x = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device), indexing="ij",
        )
        def invalid_mask(flow, occ):
            dest_x, dest_y = x + flow[0, 0], y + flow[0, 1]
            outside = (dest_x < 0) | (dest_x > width - 1) | (dest_y < 0) | (dest_y > height - 1)
            return ((occ[0] > 0) | outside).cpu().numpy().astype(np.uint8)
        return {
            "f_flow": forward[0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            "b_flow": backward[0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            "f_mask": invalid_mask(forward, f_occ),
            "b_mask": invalid_mask(backward, b_occ),
        }

    def infer_dataset_sample(self, sample):
        """Estimate adjacent flow for dataset.py's unbatched [-1,1] RGB sample.

        Return CPU tensors matching its f_flows/b_flows/f_masks/b_masks keys,
        with masks converted to valid=1. Existing sample fields are untouched.
        """
        images = sample["image"]
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("sample['image'] must have shape [T, 3, H, W]")
        if images.shape[0] < 2 or not torch.isfinite(images).all() or images.min() < -1 or images.max() > 1:
            raise ValueError("Sample requires at least two finite RGB frames in [-1, 1]")
        rgb = (images.detach().float() + 1) * 127.5
        results = [self.infer_pair(rgb[i], rgb[i + 1]) for i in range(len(rgb) - 1)]
        return {
            "f_flows": torch.from_numpy(np.stack([r["f_flow"] for r in results])).permute(0, 3, 1, 2),
            "b_flows": torch.from_numpy(np.stack([r["b_flow"] for r in results])).permute(0, 3, 1, 2),
            "f_masks": torch.from_numpy(np.stack([1 - r["f_mask"] for r in results])).unsqueeze(1).float(),
            "b_masks": torch.from_numpy(np.stack([1 - r["b_mask"] for r in results])).unsqueeze(1).float(),
        }
