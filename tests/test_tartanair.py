import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image
import torch
from zipdepth.course.data import TartanAirPairDataset, transform_pair

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

class TartanAirTests(unittest.TestCase):
    def test_flow_resize_flip_and_crop_bounds(self):
        image = torch.zeros(2, 3, 4, 6)
        depth = torch.full((2, 1, 4, 6), 2.)
        flow = torch.zeros(2, 4, 6)
        flow[0] = 1
        valid = torch.ones(1, 4, 6, dtype=torch.bool)
        rgb, inv, mask, f, fm = transform_pair(image, depth, flow, valid, 8)
        self.assertEqual(rgb.shape, (2, 3, 8, 8))
        torch.testing.assert_close(inv, torch.full_like(inv, .5))
        torch.testing.assert_close(f[0], torch.full_like(f[0], 2.))
        self.assertFalse(fm[:, :, -2:].any())
        _, _, _, flipped, flipped_mask = transform_pair(image, depth, flow, valid, 8, True)
        torch.testing.assert_close(flipped[0], torch.full_like(flipped[0], -2.))
        self.assertFalse(flipped_mask[:, :, :2].any())

    def test_invalid_depth_and_flow(self):
        d = torch.tensor([0., float("nan"), float("inf"), 90.]).repeat(2, 1, 4, 1)
        image = torch.zeros(2, 3, 4, 4)
        flow = torch.zeros(2, 4, 4)
        flow[:, 0, 0] = float("nan")
        _, target, mask, f, fm = transform_pair(image, d, flow, torch.zeros(1, 4, 4, dtype=torch.bool), 4)
        self.assertFalse(mask.any())
        self.assertFalse(fm.any())
        self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(torch.isfinite(f).all())

    def test_flow_interpolation_excludes_occluded_support(self):
        image = torch.zeros(2, 3, 2, 2)
        depth = torch.ones(2, 1, 2, 2)
        flow = torch.zeros(2, 2, 2)
        flow[:, 0, 0] = 10000
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        valid[:, 0, 0] = False
        _, _, _, resized, mask = transform_pair(image, depth, flow, valid, 4)
        self.assertFalse(mask[0, 1, 1])
        self.assertTrue(mask[0, -1, -1])
        torch.testing.assert_close(resized, torch.zeros_like(resized))

    def test_split_disjoint_reproducible(self):
        split = module("prepare_tartanair").split_trajectories
        names = [f"s{scene}/Easy/P{n:03}" for scene in range(3) for n in range(10)]
        a = split(names)
        self.assertEqual(a, split(names))
        self.assertEqual(set(sum(a.values(), [])), set(names))
        self.assertFalse(set(a["train"]) & set(a["val"]))
        self.assertFalse(set(a["val"]) & set(a["test"]))

    def test_real_format_index_loader_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for traj in range(3):
                p = root / "scene" / "Easy" / f"P{traj:03}"
                for kind in ("image_left", "depth_left", "flow"):
                    (p / kind).mkdir(parents=True)
                for frame in range(2):
                    Image.fromarray(np.full((4, 6, 3), frame * 100, np.uint8)).save(p / "image_left" / f"{frame:06}_left.png")
                    np.save(p / "depth_left" / f"{frame:06}_left_depth.npy", np.full((4, 6), 3.25, np.float32))
                np.save(p / "flow" / "000000_000001_flow.npy", np.zeros((4, 6, 2), np.float32))
                mask = np.zeros((4, 6), np.uint8)
                mask[:, 2] = 100
                np.save(p / "flow" / "000000_000001_mask.npy", mask)
            manifest = module("prepare_tartanair").build_manifest(root)
            file = root / "manifest.json"
            file.write_text(json.dumps(manifest))
            dataset = TartanAirPairDataset(file, input_size=4, augment=True)
            sample = dataset[0]
            self.assertEqual(sample["image"].shape, (2, 3, 4, 4))
            self.assertEqual(sample["target"].dtype, torch.float32)
            self.assertEqual(sample["mask"].dtype, torch.bool)
            self.assertFalse(sample["flow_mask"].all())
            torch.testing.assert_close(sample["target"], torch.full_like(sample["target"], 1 / 3.25))
            torch.testing.assert_close(sample["image"], dataset[0]["image"])
            path = root / dataset.pairs[0]["rgb0"]
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, "Failed to load"):
                dataset[0]
            manifest["splits"]["val"] = manifest["splits"]["train"]
            file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leakage"):
                TartanAirPairDataset(file)

    def test_zip_traversal_rejected(self):
        extract = module("download_tartanair").safe_extract
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract(archive, root)

    def test_parallel_ranges_preserve_existing_prefix(self):
        downloader = module("download_tartanair")
        payload = bytes(range(100))
        class Response:
            status_code = 206
            def __init__(self, start=0, end=99, head=False):
                self.start, self.end = start, end
                self.headers = {"Content-Length": "100", "ETag": '"fixed"'} if head else {"Content-Range": f"bytes {start}-{end}/100"}
            def raise_for_status(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def iter_content(self, size):
                yield payload[self.start:self.end + 1]
        def get(url, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            return Response(start, end)
        with tempfile.TemporaryDirectory() as tmp, patch.object(downloader.requests, "head", return_value=Response(head=True)), patch.object(downloader.requests, "get", side_effect=get):
            path = Path(tmp) / "sample.zip.partial"
            path.write_bytes(payload[:9])
            downloader.parallel_ranges({"path": "scene/Easy/image_left.zip", "size": 100, "sha256": "fixed"}, path, workers=2)
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(path.with_suffix(path.suffix + ".chunks").exists())

if __name__ == "__main__":
    unittest.main()
