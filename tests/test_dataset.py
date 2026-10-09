"""Behavior tests for the completed legacy video dataset and sampler."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from dataset import DepthVideoDataset, RandomScale, RandomHorizontalFlip, DistributedSamplerNoEvenlyDivisible


class DatasetTests(unittest.TestCase):
    def make_data(self, root, count=8):
        trajectory = root / "scene" / "Easy" / "P000"
        flow_trajectory = root.with_name(root.name + "_flow") / "scene" / "Easy" / "P000"
        for kind in ("image_left", "depth_left"):
            (trajectory / kind).mkdir(parents=True)
        for kind in ("f_flow", "b_flow", "f_mask", "b_mask"):
            (flow_trajectory / kind).mkdir(parents=True)
        for frame in range(count):
            stem = f"{frame:06}_left"
            Image.fromarray(np.full((16, 24, 3), frame * 20, np.uint8)).save(trajectory / "image_left" / f"{stem}.png")
            np.save(trajectory / "depth_left" / f"{stem}_depth.npy", np.full((16, 24), 2 + frame, np.float32))
            if frame < count - 1:
                flow = np.zeros((8, 12, 2), np.float32)
                flow[..., 0], flow[..., 1] = frame + 1, 0.5
                for key, value in (("f_flow", flow), ("b_flow", -flow),
                                   ("f_mask", np.zeros((8, 12), np.uint8)),
                                   ("b_mask", np.zeros((8, 12), np.uint8))):
                    np.save(flow_trajectory / key / f"{stem}.npy", value)
        return trajectory, flow_trajectory

    def test_scale_vector_units_and_preserve_inverse_depth(self):
        x = np.ones((4, 6, 5), np.float32)
        x[..., -2], x[..., -1] = 2, 4
        out = RandomScale((0.5, 0.5), 2)(x)
        self.assertEqual(out.shape, (2, 3, 5))
        np.testing.assert_allclose(out[..., :3], 1)
        np.testing.assert_allclose(out[..., -2], 1)
        np.testing.assert_allclose(out[..., -1], 2)
        np.testing.assert_allclose(x[..., -2], 2)

    def test_flip_only_horizontal_flow_components(self):
        x = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
        with patch("numpy.random.rand", return_value=1):
            out = RandomHorizontalFlip(4)(x)
        expected = x[:, ::-1].copy()
        expected[..., -4::2] *= -1
        np.testing.assert_array_equal(out, expected)
        with patch("numpy.random.rand", return_value=0):
            np.testing.assert_array_equal(RandomHorizontalFlip(0)(x), x)

    def test_sequence_split_shapes_values_directions_and_deterministic_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "custom"
            self.make_data(root)
            train = DepthVideoDataset("train", root, crop_size=8, seq_len=4)
            val = DepthVideoDataset("val", root, crop_size=8, seq_len=4)
            self.assertEqual((len(train), len(val)), (1, 1))
            self.assertTrue(train.data_paths[0][1][0][0].endswith("000000_left.png"))
            self.assertTrue(val.data_paths[0][1][0][0].endswith("000004_left.png"))
            self.assertIsNone(val.data_paths[0][1][0][3])
            self.assertIsNone(val.data_paths[0][1][-1][2])
            self.assertTrue(val.data_paths[0][1][1][3].endswith("000004_left.npy"))
            sample = val[0]
            for key, shape in (("image", (4, 3, 8, 8)), ("depth", (4, 1, 8, 8)),
                               ("f_flows", (3, 2, 8, 8)), ("b_flows", (3, 2, 8, 8)),
                               ("f_masks", (3, 1, 8, 8)), ("b_masks", (3, 1, 8, 8))):
                self.assertEqual(tuple(sample[key].shape), shape)
                self.assertEqual(sample[key].dtype, torch.float32)
                self.assertTrue(sample[key].is_contiguous())
                torch.testing.assert_close(sample[key], val[0][key])
            torch.testing.assert_close(sample["image"][0], torch.full_like(sample["image"][0], 80 / 127.5 - 1))
            torch.testing.assert_close(sample["depth"][0], torch.full_like(sample["depth"][0], 1 / 6))
            torch.testing.assert_close(sample["f_flows"][0, 0], torch.full((8, 8), 10.))
            torch.testing.assert_close(sample["b_flows"], -sample["f_flows"])
            self.assertFalse(sample["f_masks"].any())  # endpoints leave the crop
            train.scale["TartanAir"] = RandomScale((0.5, 0.5), 12, slice(16, 22))
            with patch("numpy.random.rand", return_value=1), patch("numpy.random.randint", return_value=0):
                augmented = train[0]
            self.assertEqual(augmented["image"].shape[-2:], (8, 8))
            torch.testing.assert_close(augmented["depth"][0], torch.full((1, 8, 8), .5))
            torch.testing.assert_close(augmented["f_flows"][0, 0], torch.full((8, 8), -1.))
            torch.testing.assert_close(augmented["f_flows"][0, 1], torch.full((8, 8), .5))
            self.assertFalse(augmented["f_masks"][0, :, :, 0].any())

    def test_invalid_depth_and_invalid_flow_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tartanair"
            trajectory, flow_trajectory = self.make_data(root, 2)
            depth = np.zeros((16, 24), np.float32)
            depth[0, :3] = [np.nan, np.inf, -1]
            np.save(trajectory / "depth_left" / "000000_left_depth.npy", depth)
            flow = np.full((8, 12, 2), np.nan, np.float32)
            np.save(flow_trajectory / "f_flow" / "000000_left.npy", flow)
            sample = DepthVideoDataset("val", root, 8, 2)[0]
            self.assertTrue(torch.isfinite(sample["depth"]).all())
            self.assertFalse(sample["depth"][0].any())
            self.assertTrue(torch.isfinite(sample["f_flows"]).all())
            self.assertFalse(sample["f_masks"].any())
            # Official masks may encode invalidity as 100/200 rather than 1.
            np.save(flow_trajectory / "b_mask" / "000000_left.npy", np.full((8, 12), 100, np.uint8))
            self.assertFalse(DepthVideoDataset("val", root, 8, 2)[0]["b_masks"].any())

    def test_scale_excludes_invalid_interpolation_support(self):
        x = np.zeros((2, 2, 4), np.float32)
        x[..., 1] = 1
        x[0, 0, 1] = 0
        result = RandomScale((2, 2), 2, slice(1, 2))(x)
        self.assertEqual(result.shape, (4, 4, 4))
        self.assertEqual(result[1, 1, 1], 0)
        self.assertEqual(result[-1, -1, 1], 1)
        self.assertEqual(set(np.unique(result[..., 1])), {0., 1.})

    def test_missing_flow_fails_during_indexing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "custom"
            _, flow_trajectory = self.make_data(root, 2)
            (flow_trajectory / "b_flow" / "000000_left.npy").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "estimate_flow.py"):
                DepthVideoDataset("val", root, 8, 2)

    def test_argument_validation_and_small_datasets(self):
        for arguments in (("test",), ("val", None, 0), ("val", None, 8, 1)):
            with self.assertRaises(ValueError):
                DepthVideoDataset(*arguments)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "custom"
            self.make_data(root, 2)
            self.assertEqual(len(DepthVideoDataset("train", root, 8, 2)), 0)
            self.assertEqual(len(DepthVideoDataset("val", root, 8, 2)), 1)

    def test_sampler_no_duplicates_coverage_and_short_replicas(self):
        for count in (0, 2, 10):
            dataset = list(range(count))
            samplers = [DistributedSamplerNoEvenlyDivisible(dataset, 4, rank, False) for rank in range(4)]
            lengths = [len(s) for s in samplers]
            partitions = [list(s) for s in samplers]
            self.assertEqual(sorted(sum(partitions, [])), dataset)
            self.assertEqual([len(x) for x in partitions], lengths)
            self.assertLessEqual(max(lengths) - min(lengths), 1)

    def test_sampler_epoch_reproducibility_and_argument_validation(self):
        sampler = DistributedSamplerNoEvenlyDivisible(list(range(30)), 3, 1)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))
        for replicas, rank in ((0, 0), (2, 2), (2, -1)):
            with self.assertRaises(ValueError):
                DistributedSamplerNoEvenlyDivisible([], replicas, rank)
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            DistributedSamplerNoEvenlyDivisible([])


if __name__ == "__main__":
    unittest.main()
