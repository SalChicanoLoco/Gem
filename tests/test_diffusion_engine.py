"""
Tests for PyTorchDiffusionEngine and hardware acceleration checks.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from agents.diffusion_engine import (
    MAX_INFERENCE_STEPS,
    PyTorchDiffusionEngine,
    _find_lora_weight_file,
    _is_blank,
)
from agents.lora_trainer import read_adapter_meta, write_adapter_meta
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
            self.assertLessEqual(result["steps_used"], MAX_INFERENCE_STEPS)

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


class TestPrecisionAndSafetyChecker(unittest.TestCase):
    """
    The NSFW classifier blanks benign renders to solid black, which made float16
    unusable. It is off by default; the precision it unblocks halves inference cost.
    """

    def test_mps_defaults_to_half_precision(self):
        engine = PyTorchDiffusionEngine(model_id="runwayml/stable-diffusion-v1-5")
        if engine.device == "mps":
            self.assertEqual(str(engine._inference_dtype()), "torch.float16")

    def test_models_unstable_in_half_precision_stay_float32(self):
        """
        segmind/tiny-sd renders a solid frame on most seeds in float16 on MPS,
        measured 2 of 3 against 0 of 3 in float32, so it opts out of the default.
        """
        engine = PyTorchDiffusionEngine(model_id="segmind/tiny-sd")
        if engine.device == "mps":
            self.assertEqual(str(engine._inference_dtype()), "torch.float32")

    def test_explicit_dtype_overrides_the_unstable_list(self):
        """The list is a default, not a restriction."""
        engine = PyTorchDiffusionEngine(model_id="segmind/tiny-sd", dtype="float16")
        self.assertEqual(str(engine._inference_dtype()), "torch.float16")

    def test_small_renders_drop_to_float32(self):
        """
        Half precision blanks below roughly 384px regardless of model: SD 1.5 at
        256x256, 15 steps, seed 11 renders a solid frame in float16 and a correct
        one in float32.
        """
        engine = PyTorchDiffusionEngine(model_id="runwayml/stable-diffusion-v1-5")
        if engine.device == "mps":
            self.assertEqual(str(engine._inference_dtype(256, 256)), "torch.float32")

    def test_large_renders_keep_half_precision(self):
        engine = PyTorchDiffusionEngine(model_id="runwayml/stable-diffusion-v1-5")
        if engine.device == "mps":
            self.assertEqual(str(engine._inference_dtype(512, 512)), "torch.float16")

    def test_the_smaller_edge_decides(self):
        """A wide, short frame is as numerically thin as a small square one."""
        engine = PyTorchDiffusionEngine(model_id="runwayml/stable-diffusion-v1-5")
        if engine.device == "mps":
            self.assertEqual(str(engine._inference_dtype(1024, 256)), "torch.float32")

    def test_explicit_dtype_overrides_the_size_rule(self):
        engine = PyTorchDiffusionEngine(model_id="runwayml/stable-diffusion-v1-5", dtype="float16")
        self.assertEqual(str(engine._inference_dtype(256, 256)), "torch.float16")

    def test_safety_checker_is_off_by_default(self):
        self.assertFalse(PyTorchDiffusionEngine().safety_checker)

    def test_safety_checker_can_be_opted_into(self):
        self.assertTrue(PyTorchDiffusionEngine(safety_checker=True).safety_checker)

    def test_dtype_override_is_honoured(self):
        engine = PyTorchDiffusionEngine(dtype="float32")
        self.assertEqual(str(engine._inference_dtype()), "torch.float32")

    def test_unknown_dtype_falls_back_rather_than_crashing(self):
        engine = PyTorchDiffusionEngine(dtype="nonsense")
        self.assertIn(str(engine._inference_dtype()),
                      ("torch.float16", "torch.float32"))

    def test_blank_render_is_detected(self):
        """A solid frame must not be reportable as a successful render."""
        from PIL import Image
        self.assertTrue(_is_blank(Image.new("RGB", (32, 32), (0, 0, 0))))
        self.assertTrue(_is_blank(Image.new("RGB", (32, 32), (255, 255, 255))))

    def test_detailed_render_is_not_flagged_blank(self):
        from PIL import Image
        import random
        img = Image.new("RGB", (32, 32))
        rnd = random.Random(0)
        img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                     for _ in range(32 * 32)])
        self.assertFalse(_is_blank(img))


class TestAdapterBaseModelResolution(unittest.TestCase):
    """
    A LoRA only fits the UNet it was trained on. The engine must pair the two
    without the caller having to know, and must refuse rather than quietly render
    as the base model when it cannot.
    """

    def setUp(self):
        self.engine = PyTorchDiffusionEngine(model_id="segmind/tiny-sd")
        self.tmp = tempfile.TemporaryDirectory()
        self.adapter = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def test_sidecar_roundtrips(self):
        write_adapter_meta(self.adapter, {"base_model_id": "runwayml/stable-diffusion-v1-5", "rank": 8})
        self.assertEqual(read_adapter_meta(self.adapter)["base_model_id"], "runwayml/stable-diffusion-v1-5")

    def test_missing_sidecar_reads_as_none(self):
        self.assertIsNone(read_adapter_meta(self.adapter))

    def test_declared_base_is_loaded_when_caller_names_no_model(self):
        write_adapter_meta(self.adapter, {"base_model_id": "runwayml/stable-diffusion-v1-5"})
        self.engine.lora_path = self.adapter

        self.assertEqual(self.engine._resolve_base_model(None), "runwayml/stable-diffusion-v1-5")
        self.assertIsNone(self.engine.lora_error)

    def test_explicit_model_wins_and_adapter_is_refused(self):
        """An explicit model_id is the caller's choice; the adapter yields, loudly."""
        write_adapter_meta(self.adapter, {"base_model_id": "runwayml/stable-diffusion-v1-5"})
        self.engine.lora_path = self.adapter

        self.assertEqual(self.engine._resolve_base_model("segmind/tiny-sd"), "segmind/tiny-sd")
        self.assertIsNotNone(self.engine.lora_error)
        self.assertIn("runwayml/stable-diffusion-v1-5", self.engine.lora_error)

    def test_matching_base_needs_no_switch(self):
        write_adapter_meta(self.adapter, {"base_model_id": "segmind/tiny-sd"})
        self.engine.lora_path = self.adapter

        self.assertIsNone(self.engine._resolve_base_model(None))
        self.assertIsNone(self.engine.lora_error)

    def test_adapter_without_sidecar_is_left_alone(self):
        """Unknown base is not a mismatch; the load is still attempted."""
        self.engine.lora_path = self.adapter

        self.assertIsNone(self.engine._resolve_base_model(None))
        self.assertIsNone(self.engine.lora_error)

    def test_failing_load_with_adapter_does_not_recurse_forever(self):
        """
        The tiny-sd fallback and the adapter's declared base used to fight: the
        fallback set model_id, then _resolve_base_model overrode it back to the
        adapter's base on the next call, looping until the stack blew.
        """
        write_adapter_meta(self.adapter, {"base_model_id": "runwayml/stable-diffusion-v1-5"})
        self.engine.lora_path = self.adapter
        self.engine.model_id = "models/sd15"

        with patch("agents.diffusion_engine.AutoPipelineForText2Image") as auto:
            auto.from_pretrained.side_effect = OSError("model is not cached locally")
            result = self.engine.initialize_pipeline()

        self.assertFalse(result)
        self.assertEqual(self.engine.model_id, "segmind/tiny-sd")


class TestLoraWeightFileResolution(unittest.TestCase):
    """
    diffusers refuses to guess the adapter filename when the Hub is unreachable,
    so the engine resolves it from the directory instead.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, name):
        open(os.path.join(self.d, name), "w").close()

    def test_finds_safetensors(self):
        self._touch("pytorch_lora_weights.safetensors")
        self.assertEqual(_find_lora_weight_file(self.d), "pytorch_lora_weights.safetensors")

    def test_prefers_safetensors_over_bin(self):
        self._touch("pytorch_lora_weights.bin")
        self._touch("pytorch_lora_weights.safetensors")
        self.assertEqual(_find_lora_weight_file(self.d), "pytorch_lora_weights.safetensors")

    def test_falls_back_to_bin(self):
        self._touch("pytorch_lora_weights.bin")
        self.assertEqual(_find_lora_weight_file(self.d), "pytorch_lora_weights.bin")

    def test_returns_none_when_no_weights(self):
        self._touch("adapter_meta.json")
        self.assertIsNone(_find_lora_weight_file(self.d))

    def test_missing_directory_returns_none(self):
        self.assertIsNone(_find_lora_weight_file(os.path.join(self.d, "nope")))


if __name__ == "__main__":
    unittest.main()
