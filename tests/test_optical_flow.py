"""Check the geometry and dataset conventions at the GMFlow integration boundary."""

import unittest

import numpy as np
import torch

from zipdepth.optical_flow import GMFlowEstimator


class ConstantBidirectionalModel:
    def __call__(self, first, second, **kwargs):
        flows = torch.empty((2, 2, *first.shape[-2:]), device=first.device)
        flows[0, 0], flows[0, 1] = 2, 3
        flows[1, 0], flows[1, 1] = -2, -3
        return {"flow_preds": [flows]}


class OpticalFlowGeometryTests(unittest.TestCase):
    def test_resize_restores_each_displacement_axis_and_marks_outside_pixels(self):
        # Avoid loading learned weights: exercise the integration's geometry directly.
        estimator = GMFlowEstimator.__new__(GMFlowEstimator)
        estimator.device = torch.device("cpu")
        estimator.inference_size = (30, 32)
        estimator.model = ConstantBidirectionalModel()
        estimator._consistency_check = lambda f, b: (
            torch.zeros_like(f[:, 0]), torch.zeros_like(b[:, 0])
        )
        result = estimator.infer_pair(np.zeros((90, 64, 3), np.uint8),
                                      np.zeros((90, 64, 3), np.uint8))
        np.testing.assert_allclose(result["f_flow"][..., 0], 4)
        np.testing.assert_allclose(result["f_flow"][..., 1], 9)
        np.testing.assert_allclose(result["b_flow"][..., 0], -4)
        np.testing.assert_allclose(result["b_flow"][..., 1], -9)
        self.assertTrue(result["f_mask"][-9:, :].all())
        self.assertTrue(result["f_mask"][:, -4:].all())
        self.assertFalse(result["f_mask"][:-9, :-4].any())
        self.assertTrue(result["b_mask"][:9, :].all())
        self.assertTrue(result["b_mask"][:, :4].all())

    def test_sample_conversion_preserves_time_order_and_valid_mask_convention(self):
        estimator = GMFlowEstimator.__new__(GMFlowEstimator)
        seen = []
        def record_pair(first, second):
            seen.append((first.clone(), second.clone()))
            h, w = first.shape[-2:]
            return {"f_flow": np.full((h, w, 2), len(seen), np.float32),
                    "b_flow": np.full((h, w, 2), -len(seen), np.float32),
                    "f_mask": np.zeros((h, w), np.uint8),
                    "b_mask": np.ones((h, w), np.uint8)}
        estimator.infer_pair = record_pair
        images = torch.stack([torch.full((3, 4, 6), value) for value in (-1., 0., 1.)])
        result = estimator.infer_dataset_sample({"image": images})
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0].max().item(), 0)
        self.assertEqual(seen[0][1].max().item(), 127.5)
        self.assertEqual(seen[1][1].max().item(), 255)
        self.assertEqual(tuple(result["f_flows"].shape), (2, 2, 4, 6))
        self.assertTrue(torch.equal(result["f_flows"][:, 0, 0, 0], torch.tensor([1., 2.])))
        self.assertTrue(result["f_masks"].all())
        self.assertFalse(result["b_masks"].any())
        self.assertTrue(torch.equal(images[0], torch.full((3, 4, 6), -1.)))


if __name__ == "__main__":
    unittest.main()
