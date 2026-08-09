"""
Tests for the real LoRA trainer.

The full training loop is exercised separately (it loads a multi-GB pipeline);
these cover dataset discovery, capability reporting, and the contract that a
failed run never reports a checkpoint.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from agents.lora_trainer import LoRALocalTrainer, TrainingConfig, discover_training_pairs


def _write_image(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32), (120, 60, 30)).save(path)


class TestDiscoverTrainingPairs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_caption_comes_from_subdirectory_name(self):
        _write_image(os.path.join(self.root, "roadrunner_desert", "a.png"))
        pairs = discover_training_pairs(self.root)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], "roadrunner desert")

    def test_sidecar_caption_wins(self):
        img = os.path.join(self.root, "sports_car", "a.png")
        _write_image(img)
        with open(os.path.join(self.root, "sports_car", "a.txt"), "w") as f:
            f.write("a red convertible at dusk")

        pairs = discover_training_pairs(self.root)
        self.assertEqual(pairs[0][1], "a red convertible at dusk")

    def test_uncaptioned_root_images_are_skipped(self):
        """The root directory's name is not a caption; training on it teaches noise."""
        _write_image(os.path.join(self.root, "loose.png"))
        _write_image(os.path.join(self.root, "trout", "b.png"))

        pairs = discover_training_pairs(self.root)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], "trout")

    def test_root_image_with_sidecar_is_kept(self):
        _write_image(os.path.join(self.root, "loose.png"))
        with open(os.path.join(self.root, "loose.txt"), "w") as f:
            f.write("a mountain stream")

        pairs = discover_training_pairs(self.root)
        self.assertEqual(pairs[0][1], "a mountain stream")

    def test_non_images_ignored(self):
        with open(os.path.join(self.root, "README.md"), "w") as f:
            f.write("notes")
        self.assertEqual(discover_training_pairs(self.root), [])


class TestTrainerContract(unittest.TestCase):
    def test_empty_directory_fails_without_claiming_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = LoRALocalTrainer(TrainingConfig(max_steps=1), output_dir=tmp)
            result = trainer.train(training_dir=tmp, max_steps=1)

            self.assertFalse(result["success"])
            self.assertFalse(result["trained"])
            self.assertNotIn("weights_dir", result)

    def test_missing_dependencies_report_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = LoRALocalTrainer(output_dir=tmp)
            with patch.object(
                trainer, "available",
                return_value={"can_train": False, "missing_dependency": "no torch", "device": "cpu"},
            ):
                result = trainer.train(training_dir=tmp)

            self.assertFalse(result["success"])
            self.assertFalse(result["trained"])
            self.assertIn("no torch", result["error"])

    def test_available_reports_device(self):
        trainer = LoRALocalTrainer()
        status = trainer.available()
        self.assertIn("can_train", status)
        self.assertIn(status["device"], {"mps", "cuda", "cpu"})


if __name__ == "__main__":
    unittest.main()
