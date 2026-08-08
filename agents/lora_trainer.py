"""
LoRALocalTrainer (agents/lora_trainer.py)

Real LoRA fine-tuning of a Stable Diffusion UNet on Apple Silicon (MPS).

This performs actual training: images are encoded to latents by the VAE, noise is
added on a DDPM schedule, the UNet predicts that noise, and the MSE between
prediction and target is backpropagated into LoRA adapter weights. Reported
losses come from those tensors. Weights are written as safetensors and can be
loaded back into a diffusion pipeline.

Captions are taken from a per-image ``.txt`` sidecar when present, otherwise from
the containing directory name (``training_data/roadrunner_desert`` -> "roadrunner
desert"), which matches how the training assets in this repo are organised.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
TRAINING_DEPS_AVAILABLE = False
MISSING_DEPENDENCY: Optional[str] = None

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError as e:  # pragma: no cover - environment dependent
    MISSING_DEPENDENCY = str(e)

try:
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from transformers import CLIPTextModel, CLIPTokenizer

    TRAINING_DEPS_AVAILABLE = TORCH_AVAILABLE
except ImportError as e:  # pragma: no cover - environment dependent
    MISSING_DEPENDENCY = str(e)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class TrainingConfig:
    """Hyperparameters for a LoRA run."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 512
    rank: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 200
    batch_size: int = 1
    seed: int = 0
    # Attention projections are the standard LoRA injection points for SD UNets.
    target_modules: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")


def discover_training_pairs(training_dir: str) -> List[Tuple[str, str]]:
    """
    Walk a training directory and pair each image with a caption.

    A ``<image>.txt`` sidecar wins. Otherwise the containing subdirectory name is
    used, with underscores turned into spaces.

    Images sitting directly in the training root are skipped unless they have a
    sidecar: the root directory's own name ("training data") is not a caption,
    and training on it would teach the adapter that phrase. Skipped files are
    logged so they are not lost silently.
    """
    training_dir = os.path.normpath(training_dir)
    pairs: List[Tuple[str, str]] = []
    skipped: List[str] = []

    for root, _dirs, files in os.walk(training_dir):
        for name in sorted(files):
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image_path = os.path.join(root, name)

            sidecar = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    caption = f.read().strip()
            elif os.path.normpath(root) == training_dir:
                skipped.append(image_path)
                continue
            else:
                caption = os.path.basename(root).replace("_", " ").replace("-", " ").strip()

            if caption:
                pairs.append((image_path, caption))

    if skipped:
        logger.warning(
            "Skipped %d uncaptioned image(s) in the training root: %s. "
            "Move each into a subdirectory named for its subject, or add a .txt sidecar.",
            len(skipped),
            ", ".join(os.path.basename(p) for p in skipped),
        )
    return pairs


if TORCH_AVAILABLE:

    class _CaptionedImageDataset(Dataset):
        """Images resized to a square resolution, normalised to [-1, 1], with token ids."""

        def __init__(self, pairs: List[Tuple[str, str]], tokenizer, resolution: int):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.resolution = resolution

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            image_path, caption = self.pairs[index]
            image = Image.open(image_path).convert("RGB").resize(
                (self.resolution, self.resolution), Image.BICUBIC
            )

            pixels = torch.from_numpy(_to_float_array(image))
            pixels = pixels.permute(2, 0, 1) / 127.5 - 1.0

            tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            return {"pixel_values": pixels, "input_ids": tokens.input_ids[0]}


def _to_float_array(image) -> Any:
    """Convert a PIL image to a float32 HWC array without importing numpy at module scope."""
    import numpy as np

    return np.array(image, dtype="float32")


class LoRALocalTrainer:
    """
    Trains LoRA adapters for a Stable Diffusion UNet against local images.

    Every number this class reports is measured. If the training dependencies are
    missing it fails loudly rather than returning a plausible-looking result.
    """

    def __init__(self, config: Optional[TrainingConfig] = None, output_dir: str = "checkpoints"):
        self.config = config or TrainingConfig()
        self.output_dir = output_dir

    def available(self) -> Dict[str, Any]:
        """Report whether real training can run here, and why not if it cannot."""
        device = "cpu"
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        return {
            "can_train": TRAINING_DEPS_AVAILABLE,
            "missing_dependency": None if TRAINING_DEPS_AVAILABLE else MISSING_DEPENDENCY,
            "device": device,
        }

    def train(
        self,
        training_dir: str = "static/training_data",
        max_steps: Optional[int] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Run a LoRA fine-tune and save the adapter weights.

        Args:
            training_dir: Directory of training images.
            max_steps: Override the configured step count.
            progress_callback: Optional callable receiving each step's log dict.

        Returns:
            A dict with measured per-step losses and the saved weight path, or an
            error describing why training could not run.
        """
        status = self.available()
        if not status["can_train"]:
            return {
                "success": False,
                "error": f"Training dependencies unavailable: {status['missing_dependency']}",
                "trained": False,
            }

        pairs = discover_training_pairs(training_dir)
        if not pairs:
            return {
                "success": False,
                "error": f"No training images found under {training_dir}",
                "trained": False,
            }

        cfg = self.config
        steps = max_steps if max_steps is not None else cfg.max_steps
        device = status["device"]
        # float32 throughout: MPS autograd is unreliable in half precision.
        dtype = torch.float32

        logger.info("Loading %s components for LoRA training on %s", cfg.model_id, device)
        tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(cfg.model_id, subfolder="text_encoder", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(cfg.model_id, subfolder="vae", torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(cfg.model_id, subfolder="unet", torch_dtype=dtype)
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")

        # Only the LoRA adapters learn; the base model stays frozen.
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        unet.add_adapter(
            LoraConfig(
                r=cfg.rank,
                lora_alpha=cfg.rank,
                init_lora_weights="gaussian",
                target_modules=list(cfg.target_modules),
            )
        )

        trainable = [p for p in unet.parameters() if p.requires_grad]
        if not trainable:
            return {
                "success": False,
                "error": "LoRA adapter injection produced no trainable parameters",
                "trained": False,
            }

        vae.to(device)
        text_encoder.to(device)
        unet.to(device)

        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
        dataset = _CaptionedImageDataset(pairs, tokenizer, cfg.resolution)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

        unet.train()
        step_logs: List[Dict[str, Any]] = []
        start_time = time.time()
        step = 0

        while step < steps:
            for batch in loader:
                if step >= steps:
                    break

                pixel_values = batch["pixel_values"].to(device, dtype=dtype)
                input_ids = batch["input_ids"].to(device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn(latents.shape, generator=generator).to(device)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), generator=generator
                ).to(device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                log = {"step": step, "loss": round(loss.detach().item(), 6)}
                step_logs.append(log)
                if progress_callback:
                    progress_callback(log)
                logger.info("LoRA step %d/%d loss=%.6f", step, steps, log["loss"])

        elapsed = round(time.time() - start_time, 2)

        os.makedirs(self.output_dir, exist_ok=True)
        run_id = f"lora_sd15_r{cfg.rank}_{int(time.time())}"
        weight_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(weight_dir, exist_ok=True)

        lora_state = get_peft_model_state_dict(unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=weight_dir, unet_lora_layers=lora_state, safe_serialization=True
        )

        losses = [entry["loss"] for entry in step_logs]
        return {
            "success": True,
            "trained": True,
            "run_id": run_id,
            "weights_dir": weight_dir,
            "device": device,
            "model_id": cfg.model_id,
            "steps_completed": step,
            "images_used": len(pairs),
            "rank": cfg.rank,
            "learning_rate": cfg.learning_rate,
            "resolution": cfg.resolution,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "mean_loss": round(sum(losses) / len(losses), 6),
            "elapsed_seconds": elapsed,
            "step_history": step_logs,
        }
