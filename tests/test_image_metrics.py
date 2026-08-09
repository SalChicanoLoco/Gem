"""Tests for real image similarity measurement."""

import os
import tempfile
import unittest

from PIL import Image

from agents.image_metrics import compare, histogram_correlation, ssim


def _write(path: str, color, size=(128, 128)) -> str:
    Image.new("RGB", size, color).save(path)
    return path


def _write_gradient(path: str, size=(128, 128)) -> str:
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (x * 2 % 256, y * 2 % 256, (x + y) % 256)
    img.save(path)
    return path


class TestImageMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_images_score_100(self):
        a = _write_gradient(os.path.join(self.d, "a.png"))
        result = compare(a, a)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["ssim"], 1.0, places=3)
        self.assertEqual(result["score"], 100.0)

    def test_identical_scores_above_different(self):
        """The metric must actually discriminate, not just return a constant."""
        a = _write_gradient(os.path.join(self.d, "a.png"))
        b = _write(os.path.join(self.d, "b.png"), (10, 10, 10))

        same = compare(a, a)["score"]
        different = compare(a, b)["score"]
        self.assertGreater(same, different)

    def test_ssim_is_bounded(self):
        a = _write(os.path.join(self.d, "a.png"), (255, 0, 0))
        b = _write(os.path.join(self.d, "b.png"), (0, 0, 255))
        value = ssim(a, b)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_histogram_correlation_detects_matching_palette(self):
        a = _write(os.path.join(self.d, "a.png"), (120, 60, 30))
        b = _write(os.path.join(self.d, "b.png"), (120, 60, 30))
        c = _write(os.path.join(self.d, "c.png"), (0, 200, 255))
        self.assertGreater(histogram_correlation(a, b), histogram_correlation(a, c))

    def test_missing_file_reports_failure(self):
        a = _write(os.path.join(self.d, "a.png"), (1, 2, 3))
        result = compare(a, os.path.join(self.d, "nope.png"))
        self.assertFalse(result["success"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
