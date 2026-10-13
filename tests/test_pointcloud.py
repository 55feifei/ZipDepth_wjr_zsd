"""Check the geometry of turning relative inverse depth into point clouds."""

import unittest

import numpy as np

from zipdepth.utils.pointcloud import (InverseDepthMapper, depth_to_points,
                                       intrinsics_from_fov, scale_intrinsics)


class PointCloudGeometryTests(unittest.TestCase):
    def test_backprojection_reprojects_to_the_source_pixels(self):
        rng = np.random.default_rng(0)
        depth = rng.uniform(1, 5, (6, 8)).astype(np.float32)
        intrinsics = (7.0, 9.0, 3.5, 2.5)
        points, _ = depth_to_points(depth, intrinsics)
        fx, fy, cx, cy = intrinsics
        u = points[:, 0] * fx / points[:, 2] + cx
        v = points[:, 1] * fy / points[:, 2] + cy
        vv, uu = np.mgrid[0:6, 0:8]
        np.testing.assert_allclose(u, uu.ravel(), atol=1e-4)
        np.testing.assert_allclose(v, vv.ravel(), atol=1e-4)
        np.testing.assert_allclose(points[:, 2], depth.ravel())

    def test_stride_keeps_colours_aligned_and_drops_invalid_depth(self):
        depth = np.ones((4, 4), np.float32)
        depth[0, 2] = np.nan
        depth[2, 0] = 0
        colors = np.zeros((4, 4, 3), np.uint8)
        colors[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
        points, point_colors = depth_to_points(depth, (1, 1, 0, 0), colors, stride=2)
        self.assertEqual(len(points), 2)
        np.testing.assert_allclose(point_colors[:, 0] * 255, [0, 100], atol=1e-3)
        np.testing.assert_allclose(points[:, :2], [[0, 0], [2, 2]])

    def test_edge_filter_removes_both_sides_of_a_depth_jump_only(self):
        depth = np.ones((3, 6), np.float32)
        depth[:, 3:] = 4
        points, _ = depth_to_points(depth, (1, 1, 0, 0), edge_threshold=0.1)
        kept_columns = sorted(set(np.round(points[:, 0] / points[:, 2]).astype(int)))
        self.assertEqual(kept_columns, [0, 1, 4, 5])
        all_points, _ = depth_to_points(depth, (1, 1, 0, 0), edge_threshold=0)
        self.assertEqual(len(all_points), depth.size)

    def test_mapper_sends_large_inverse_depth_near_and_clamps_to_range(self):
        inverse = np.linspace(0, 1, 101, dtype=np.float32).reshape(1, -1)
        depth = InverseDepthMapper(near=1, far=10, percentiles=(0, 100))(inverse)
        self.assertAlmostEqual(float(depth[0, -1]), 1.0, places=5)
        self.assertAlmostEqual(float(depth[0, 0]), 10.0, places=4)
        self.assertTrue(np.all(np.diff(depth[0]) < 0))
        clipped = InverseDepthMapper(near=2, far=5)(inverse * 100 - 3)
        self.assertGreaterEqual(float(clipped.min()), 2 - 1e-5)
        self.assertLessEqual(float(clipped.max()), 5 + 1e-4)

    def test_mapper_is_affine_invariant_and_ema_damps_range_changes(self):
        inverse = np.random.default_rng(1).uniform(0.2, 3, (10, 10)).astype(np.float32)
        np.testing.assert_allclose(InverseDepthMapper()(inverse),
                                   InverseDepthMapper()(inverse * 4 + 7), rtol=1e-4)
        mapper = InverseDepthMapper(percentiles=(0, 100), ema=0.25)
        mapper(np.array([[0, 1]], np.float32))
        mapper(np.array([[0, 5]], np.float32))
        np.testing.assert_allclose(mapper.bounds, [0, 2])

    def test_intrinsics_from_fov_and_rescaling(self):
        fx, fy, cx, cy = intrinsics_from_fov(640, 480, 90)
        self.assertAlmostEqual(fx, 320)
        self.assertEqual((cx, cy), (319.5, 239.5))
        half = scale_intrinsics((fx, fy, cx, cy), 0.5, 0.5)
        np.testing.assert_allclose(half, (160, 160, 159.5, 119.5))
        with self.assertRaises(ValueError):
            intrinsics_from_fov(640, 480, 180)


if __name__ == '__main__':
    unittest.main()
