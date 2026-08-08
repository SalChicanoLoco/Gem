"""
Tests for PyTorchDiffusionEngine and hardware acceleration checks.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from agents.diffusion_engine import PyTorchDiffusionEngine
from agents import ImageAgent


class TestPyTorchDiffusionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PyTorchDiffusionEngine()

    def test_hardware_status(self):
        status = self.engine.get_hardware_status()
        self.assertIn("torch_available", status)
        self.assertIn("mps_available", status)
        self.assertIn("active_device", status)

    def test_generation_returns_image(self):
        result = self.engine.generate("A serene mountain landscape", width=256, height=256, num_inference_steps=1)
        self.assertTrue(result.get("success"))
        self.assertIn("dimensions", result)
        self.assertEqual(result["dimensions"]["width"], 256)
        self.assertIn("base64", result)
        self.assertIn("placeholder", result)

    def test_placeholder_is_flagged_and_not_called_a_render(self):
        """A gradient placeholder must never present itself as a diffusion render."""
        with patch.object(self.engine, "initialize_pipeline", return_value=False):
            result = self.engine.generate("anything", width=64, height=64)

        self.assertTrue(result["placeholder"])
        self.assertIn("placeholder", result["engine"].lower())
        self.assertNotIn("photorealistic", result["engine"].lower())

    def test_steps_are_clamped_and_reported(self):
        result = self.engine.generate("a cube", width=64, height=64, num_inference_steps=500)
        if not result.get("placeholder"):
            self.assertEqual(result["steps_requested"], 500)
            self.assertLessEqual(result["steps_used"], 19)

    def test_image_agent_integration(self):
        agent = ImageAgent()
        with patch.object(agent.diffusion_engine, "generate") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "engine": "PyTorch MPS (Apple Silicon Metal GPU)",
                "dimensions": {"width": 256, "height": 256},
                "filepath": "static/videos/test.png",
                "relative_url": "/static/videos/test.png",
                "base64": "mock_b64",
            }
            result = agent.generate_image("A futuristic city", width=256, height=256, style="local_mps")
            self.assertTrue(result.get("success"))


if __name__ == "__main__":
    unittest.main()
