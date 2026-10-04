"""Tests for real geometric/error invariants and reproducible accumulation."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from zipdepth.course.losses import TemporalDepthLoss, flow_warp_next, masked_normalize
from zipdepth.course.training import train_temporal, validate
from zipdepth.course.training import freeze_bn_statistics, strict_load_mobile_model, enable_deterministic_mobile_pooling


class TemporalLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.target = torch.rand(1, 2, 1, 8, 8) + 0.1
        self.mask = torch.ones_like(self.target, dtype=torch.bool)
        self.flow = torch.zeros(1, 2, 8, 8)
        self.flow_mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        self.criterion = TemporalDepthLoss(0.3)

    def test_real_depth_changes_are_not_penalized(self):
        # Targets genuinely differ between frames; a shared affine-perfect
        # prediction has zero spatial and temporal error despite that change.
        pred = (self.target * 3.0 + 0.7).requires_grad_()
        loss, metrics = self.criterion(pred, self.target, self.mask, self.flow, self.flow_mask)
        self.assertLess(loss.item(), 2e-5)
        self.assertLess(metrics['temporal'].item(), 2e-5)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_framewise_scale_drift_is_visible(self):
        target = self.target[:, :1].repeat(1, 2, 1, 1, 1)
        pred = target.clone()
        pred[:, 1] *= 2
        _, metrics = self.criterion(pred, target, self.mask, self.flow, self.flow_mask)
        self.assertLess(metrics['spatial'].item(), 1e-5)
        self.assertGreater(metrics['temporal'].item(), 0.1)

    def test_forward_flow_samples_next_frame(self):
        image = torch.arange(16.0).view(1, 1, 4, 4)
        flow = torch.zeros(1, 2, 4, 4)
        flow[:, 0] = 1
        warped, inside = flow_warp_next(image, flow)
        torch.testing.assert_close(warped[..., :-1], image[..., 1:], atol=1e-6, rtol=1e-6)
        self.assertTrue(inside[..., :-1].all())
        self.assertFalse(inside[..., -1].any())

    def test_fractional_flow_matches_grid_sample_values_and_gradient(self):
        value = torch.randn(1, 2, 8, 8, requires_grad=True)
        flow = torch.randn(1, 2, 8, 8) * 0.5
        actual, _ = flow_warp_next(value, flow)
        yy, xx = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing='ij')
        grid = torch.stack((2 * (xx + flow[:, 0]) / 7 - 1, 2 * (yy + flow[:, 1]) / 7 - 1), -1)
        expected = torch.nn.functional.grid_sample(value, grid, align_corners=True)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        actual_gradient = torch.autograd.grad(actual.sum(), value, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), value)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-6)

    def test_empty_and_occluded_masks_are_safe(self):
        pred = self.target.clone().requires_grad_()
        loss, metrics = self.criterion(pred, self.target, torch.zeros_like(self.mask), self.flow, self.flow_mask)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad.abs().sum().item(), 0)
        _, metrics = self.criterion(pred.detach() * 1.3, self.target, self.mask, self.flow,
                                    torch.zeros_like(self.flow_mask))
        self.assertEqual(metrics['temporal'].item(), 0)

    def test_invalid_nan_is_removed_before_reduction(self):
        value = torch.rand(1, 1, 4, 4)
        value[..., 0, 0] = float('nan')
        mask = torch.ones_like(value, dtype=torch.bool)
        mask[..., 0, 0] = False
        self.assertTrue(torch.isfinite(masked_normalize(value, mask)).all())

    def test_nonzero_temporal_loss_has_nonzero_finite_gradient(self):
        pred = (self.target + 0.05 * torch.randn_like(self.target)).requires_grad_()
        loss, metrics = self.criterion(pred, self.target, self.mask, self.flow, self.flow_mask)
        loss.backward()
        self.assertGreater(metrics['temporal'].item(), 0)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0)


class _ToyPairs(Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, index):
        # Global RNG mimics random augmentation; checkpoint must preserve it.
        image = torch.rand(2, 3, 8, 8)
        target = image[:, :1] * 0.3 + 0.2 + index * 0.01
        return {'image': image, 'target': target,
                'mask': torch.ones_like(target, dtype=torch.bool),
                'flow': torch.zeros(2, 8, 8), 'flow_mask': torch.ones(1, 8, 8, dtype=torch.bool),
                'trajectory': 'toy', 'frame_id': index}


class EvaluationRegressionTests(unittest.TestCase):
    def test_subpixel_warp_rejects_invalid_neighbor_support(self):
        import numpy as np
        from zipdepth.course.evaluation import flow_error_change
        depth = np.ones((4, 4), np.float32)
        source_valid = np.zeros((4, 4), bool)
        source_valid[1, 1] = True
        next_valid = source_valid.copy()
        flow = np.zeros((2, 4, 4), np.float32)
        flow[0] = 0.01
        error, count = flow_error_change(depth, depth, depth, depth, source_valid, next_valid,
                                          flow, np.ones((4, 4), bool), 1.0, 0.0)
        self.assertEqual(count, 0)
        self.assertEqual(error, 0.0)

    def test_global_alignment_keeps_framewise_scale_drift_visible(self):
        import numpy as np
        from zipdepth.course.evaluation import flow_error_change
        target = np.arange(1, 17, dtype=np.float32).reshape(4, 4) / 20
        valid = np.ones((4, 4), bool)
        flow = np.zeros((2, 4, 4), np.float32)
        error, count = flow_error_change(target, target * 2, target, target, valid, valid,
                                          flow, valid, 1.0, 0.0)
        self.assertEqual(count, 16)
        self.assertGreater(error, 0.1)


class TrainingResumeTests(unittest.TestCase):
    def test_optimizer_boundary_resume_matches_uninterrupted_training(self):
        torch.manual_seed(8)
        initial = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                                nn.ReLU(), nn.Dropout2d(0.2), nn.Conv2d(4, 1, 1), nn.Softplus())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'device': 'cpu', 'amp': False, 'seed': 42, 'lr': 1e-4,
                      'temporal_weight': 0.1, 'grad_accum': 2, 'max_updates': 2,
                      'epochs': 2, 'validate_every': 1, 'max_val_pairs': 2, 'grad_scales': 2}
            uninterrupted = copy.deepcopy(initial)
            full = train_temporal({**config, 'output': str(root / 'full')}, _ToyPairs(), _ToyPairs(), uninterrupted)
            split = copy.deepcopy(initial)
            train_temporal({**config, 'max_updates': 1, 'output': str(root / 'part')}, _ToyPairs(), _ToyPairs(), split)
            resumed = copy.deepcopy(initial)
            summary = train_temporal({**config, 'output': str(root / 'part'),
                                      'resume': str(root / 'part' / 'last.pth')}, _ToyPairs(), _ToyPairs(), resumed)
            self.assertEqual(summary['updates'], full['updates'])
            for name, parameter in uninterrupted.state_dict().items():
                torch.testing.assert_close(parameter, resumed.state_dict()[name], atol=0, rtol=0)
            # The frozen BN running statistics must remain those of initialization.
            torch.testing.assert_close(resumed[1].running_mean, initial[1].running_mean)
            saved = torch.load(root / 'part' / 'last.pth', weights_only=True)
            self.assertEqual(saved['next_microbatch'], 4)

    def test_noop_resume_into_new_output_preserves_last_and_best(self):
        model = nn.Sequential(nn.Conv2d(3, 1, 1), nn.Softplus())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'device': 'cpu', 'amp': False, 'seed': 42, 'temporal_weight': 0.1,
                      'grad_accum': 2, 'max_updates': 1, 'epochs': 1, 'max_val_pairs': 1}
            train_temporal({**config, 'output': str(root / 'first')}, _ToyPairs(), _ToyPairs(), copy.deepcopy(model))
            summary = train_temporal({**config, 'output': str(root / 'new'), 'resume': str(root / 'first' / 'last.pth')},
                                     _ToyPairs(), _ToyPairs(), copy.deepcopy(model))
            self.assertTrue((root / 'new' / 'last.pth').is_file())
            self.assertTrue((root / 'new' / 'best.pth').is_file())
            self.assertEqual(summary['updates'], 1)
            self.assertEqual(summary['resume_best_status'], 'copied_source_best')
            saved = torch.load(root / 'new' / 'last.pth', weights_only=True)
            self.assertEqual(saved['global_step'], 1)

    def test_validation_rejects_all_empty_depth(self):
        class EmptyPairs(_ToyPairs):
            def __getitem__(self, index):
                pair = super().__getitem__(index)
                pair['mask'].zero_()
                return pair
        model = nn.Sequential(nn.Conv2d(3, 1, 1), nn.Softplus())
        with self.assertRaisesRegex(ValueError, 'no usable depth'):
            validate(model, EmptyPairs(), TemporalDepthLoss(), torch.device('cpu'), max_pairs=2)

    def test_resume_rejects_changed_manifest_at_same_path(self):
        model = nn.Sequential(nn.Conv2d(3, 1, 1), nn.Softplus())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'manifest.json'
            manifest.write_text('{"version": 1}', encoding='utf-8')
            config = {'device': 'cpu', 'amp': False, 'seed': 42, 'temporal_weight': 0.1,
                      'grad_accum': 2, 'max_updates': 1, 'epochs': 1, 'max_val_pairs': 1,
                      'manifest': str(manifest), 'output': str(root / 'run')}
            train_temporal(config, _ToyPairs(), _ToyPairs(), copy.deepcopy(model))
            saved = torch.load(root / 'run' / 'last.pth', weights_only=True)
            effective = json.loads((root / 'run' / 'effective_config.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['config']['manifest_sha256'], effective['manifest_sha256'])
            self.assertEqual(len(effective['manifest_sha256']), 64)
            manifest.write_text('{"version": 2}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'manifest_sha256'):
                train_temporal({**config, 'resume': str(root / 'run' / 'last.pth')},
                               _ToyPairs(), _ToyPairs(), copy.deepcopy(model))


class _DeterministicPairs(Dataset):
    def __init__(self, size=64):
        self.size = size

    def __len__(self):
        return 4

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(100 + index)
        image = torch.rand(2, 3, self.size, self.size, generator=generator)
        target = image[:, :1] * 0.3 + 0.2
        return {'image': image, 'target': target, 'mask': torch.ones_like(target, dtype=torch.bool),
                'flow': torch.zeros(2, self.size, self.size),
                'flow_mask': torch.ones(1, self.size, self.size, dtype=torch.bool),
                'trajectory': 'gpu_smoke', 'frame_id': index}


@unittest.skipUnless(os.environ.get('ZIPDEPTH_GPU_SMOKE') == '1' and torch.cuda.is_available(),
                     'Set ZIPDEPTH_GPU_SMOKE=1 to run real pretrained CUDA smoke tests')
class RealMobileGpuTests(unittest.TestCase):
    def test_real_model_384_bf16_forward_backward(self):
        model = strict_load_mobile_model('checkpoints/zipdepth_base_npu.pth', 'cuda')
        enable_deterministic_mobile_pooling(model)
        model.train()
        freeze_bn_statistics(model)
        batch = _DeterministicPairs(384)[0]
        image = batch['image'].cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(image).unsqueeze(0)
        criterion = TemporalDepthLoss(0.1)
        loss, metrics = criterion(pred, batch['target'].cuda().unsqueeze(0),
                                  batch['mask'].cuda().unsqueeze(0), batch['flow'].cuda().unsqueeze(0),
                                  batch['flow_mask'].cuda().unsqueeze(0))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        print(f'384 BF16 peak allocated MiB: {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f}')

    def test_real_cuda_bf16_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'device': 'cuda', 'amp': True, 'seed': 42, 'lr': 1e-5,
                      'temporal_weight': 0.1, 'grad_accum': 2, 'max_updates': 2,
                      'checkpoint': 'checkpoints/zipdepth_base_npu.pth', 'epochs': 2,
                      'validate_every': 1, 'max_val_pairs': 1, 'input_size': 64}
            train_temporal({**config, 'output': str(root / 'full')}, _DeterministicPairs(), _DeterministicPairs())
            train_temporal({**config, 'max_updates': 1, 'output': str(root / 'part')}, _DeterministicPairs(), _DeterministicPairs())
            train_temporal({**config, 'output': str(root / 'part'), 'resume': str(root / 'part' / 'last.pth')},
                           _DeterministicPairs(), _DeterministicPairs())
            full = torch.load(root / 'full' / 'last.pth', map_location='cpu', weights_only=True)
            resumed = torch.load(root / 'part' / 'last.pth', map_location='cpu', weights_only=True)
            for name, parameter in full['model_state_dict'].items():
                torch.testing.assert_close(parameter, resumed['model_state_dict'][name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
