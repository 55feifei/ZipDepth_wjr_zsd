"""Check the bokeh renderer and keep the Android golden fixture in sync.

Regenerate the fixture after changing zipdepth/bokeh.py:
    python tests/test_bokeh.py --write
"""

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

from zipdepth.bokeh import (FocusTracker, defocus_map, focus_disparity, normalize_disparity,
                            render_bokeh, upsample_bilinear)

FIXTURE = Path(__file__).parent / 'fixtures' / 'bokeh.json'
SIZE = 16
STRENGTH = 0.1875  # defocus radius of 3 px for a full disparity range at 16 x 16


def fixture_scene():
    """Textured 16x16 RGB with a near left half and a far, sloping right half."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    checker = ((x // 2 + y // 2) % 2).astype(np.float32)
    rgb = np.stack([0.15 + 0.7 * checker,
                    0.5 + 0.4 * np.sin(x * 0.9) * np.cos(y * 0.7),
                    0.1 + 0.8 * (x + y) / (2 * SIZE - 2)], axis=-1).astype(np.float32)
    disparity = np.where(x < SIZE // 2, 0.9, 0.35 - 0.25 * (x - SIZE // 2) / (SIZE // 2 - 1))
    disparity[5:11, 3:6] = 1.0
    return rgb, disparity.astype(np.float32)


def fixture_inverse_depths():
    _, disparity = fixture_scene()
    rng = np.random.default_rng(7)
    frames = [disparity * 3 + 1, disparity * 2.5 + 1.4, disparity * 4 + 0.5]
    frames = [f + rng.normal(0, 0.01, f.shape).astype(np.float32) for f in frames]
    frames[1][0, 0] = 40.0  # an outlier the 2/98 percentiles must ignore
    return [f.astype(np.float32) for f in frames]


def chw(image):
    return np.ascontiguousarray(image.transpose(2, 0, 1)).ravel()


def floats(array):
    return [round(float(v), 7) for v in np.asarray(array).ravel()]


def build_fixture():
    rgb, disparity = fixture_scene()
    cases = []
    for name, point, gamma, downsample in [('near_focus_full', (0.25, 0.5), 4.0, 1),
                                           ('far_focus_full', (0.9, 0.3), 4.0, 1),
                                           ('near_focus_mobile', (0.25, 0.5), 4.0, 2),
                                           ('mid_focus_mobile_gamma22', (0.6, 0.6), 2.2, 2)]:
        focus = focus_disparity(disparity, point)
        expected = render_bokeh(rgb, disparity, focus, STRENGTH, gamma, downsample)
        cases.append({'name': name, 'point': list(point), 'focus': round(focus, 7),
                      'gamma': gamma, 'downsample': downsample, 'expected': floats(chw(expected))})
    tracker, frames = FocusTracker(), []
    for index, inverse in enumerate(fixture_inverse_depths()):
        point, snap = ((0.25, 0.5), False) if index < 2 else ((0.9, 0.3), True)
        smoothed, focus = tracker.update(inverse, point, snap)
        frames.append({'inverse_depth': floats(inverse), 'point': list(point), 'snap': snap,
                       'disparity': floats(smoothed), 'focus': round(focus, 7)})
    return {'height': SIZE, 'width': SIZE, 'strength': STRENGTH,
            'rgb': floats(chw(rgb)), 'disparity': floats(disparity),
            'cases': cases, 'tracker': frames}


class BokehTests(unittest.TestCase):
    def test_uniform_image_is_unchanged(self):
        image = np.full((12, 12, 3), 0.4, np.float32)
        disparity = np.random.default_rng(0).uniform(0, 1, (12, 12)).astype(np.float32)
        for downsample in (1, 2):
            out = render_bokeh(image, disparity, 0.5, strength=0.3, downsample=downsample)
            np.testing.assert_allclose(out, image, atol=1e-5)

    def test_output_stays_inside_input_range(self):
        rng = np.random.default_rng(1)
        image = rng.uniform(0.2, 0.8, (16, 16, 3)).astype(np.float32)
        disparity = rng.uniform(0, 1, (16, 16)).astype(np.float32)
        for downsample in (1, 2):
            out = render_bokeh(image, disparity, 0.3, strength=0.2, downsample=downsample)
            self.assertGreaterEqual(float(out.min()), 0.2 - 1e-5)
            self.assertLessEqual(float(out.max()), 0.8 + 1e-5)

    def test_focused_region_stays_sharp_and_far_region_blurs(self):
        rgb, disparity = fixture_scene()
        focus = focus_disparity(disparity, (0.25, 0.5))
        out = render_bokeh(rgb, disparity, focus, STRENGTH)
        near = (disparity == np.float32(0.9))
        near[:, 6:] = False  # keep away from the depth edge, which far pixels may cover
        near_error = np.abs(out[near] - rgb[near]).mean()
        far_error = np.abs(out[:, 11:] - rgb[:, 11:]).mean()
        self.assertLess(near_error, 0.01)
        self.assertGreater(far_error, 10 * near_error)
        far = np.s_[:, 11:]
        self.assertLess(out[far].std(axis=(0, 1)).mean(), 0.6 * rgb[far].std(axis=(0, 1)).mean())

    def test_mobile_variant_keeps_in_focus_pixels_exactly(self):
        rgb, disparity = fixture_scene()
        focus = focus_disparity(disparity, (0.25, 0.5))
        out = render_bokeh(rgb, disparity, focus, STRENGTH, downsample=2)
        sharp = np.abs(defocus_map(disparity, focus, STRENGTH)) <= 0.5
        self.assertTrue(sharp.any())
        np.testing.assert_array_equal(out[sharp], rgb[sharp])

    def test_bilinear_upsampling_matches_opencv(self):
        x = np.random.default_rng(2).uniform(0, 1, (5, 7, 3)).astype(np.float32)
        expected = cv2.resize(x, (14, 10), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(upsample_bilinear(x, 2), expected, atol=1e-5)

    def test_focus_uses_window_median_and_normalisation_ignores_outliers(self):
        disparity = np.zeros((20, 20), np.float32)
        disparity[:, 10:] = 1.0
        self.assertEqual(focus_disparity(disparity, (0.0, 0.0)), 0.0)
        self.assertEqual(focus_disparity(disparity, (1.0, 1.0)), 1.0)
        inverse = np.linspace(1, 2, 400, dtype=np.float32).reshape(20, 20)
        inverse[0, 0] = 1000
        normalized, bounds = normalize_disparity(inverse)
        self.assertLess(bounds[1], 3)
        self.assertEqual(float(normalized.max()), 1.0)
        self.assertEqual(float(normalized.min()), 0.0)

    def test_fixture_matches_python_renderer(self):
        stored = json.loads(FIXTURE.read_text())
        fresh = build_fixture()
        for old, new in zip(stored['cases'] + stored['tracker'], fresh['cases'] + fresh['tracker']):
            for key in ('expected', 'disparity'):
                if key in old:
                    np.testing.assert_allclose(old[key], new[key], atol=1e-5)
            self.assertAlmostEqual(old['focus'], new['focus'], places=5)


if __name__ == '__main__':
    if '--write' in sys.argv:
        FIXTURE.write_text(json.dumps(build_fixture(), separators=(',', ':')))
        print(f'Wrote {FIXTURE}')
    else:
        unittest.main()
