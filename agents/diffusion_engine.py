"""
PyTorchDiffusionEngine - Apple Silicon Hardware-Accelerated Local Diffusion Engine.

Uses PyTorch with Metal Performance Shaders (MPS) and HuggingFace Diffusers for
fast local text-to-image synthesis on the Apple Silicon Metal GPU.

Note: the MPS backend targets the Metal GPU, not the Apple Neural Engine. The ANE
is only reachable through Core ML, which this engine does not use.
"""

import base64
import hashlib
import io
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Upper bound on denoising steps, capped to keep MPS runs inside a responsive
# interactive budget. Requests above this are clamped and reported back.
MAX_INFERENCE_STEPS = 19

MINIMAL_NEGATIVE_PROMPT = "blurry, lowres, distorted"

DEFAULT_NEGATIVE_PROMPT = (
    "extra limbs, extra heads, bad anatomy, deformed limbs, disfigured, mutated hands, "
    "bad proportions, distorted faces, missing limbs, blurry, lowres, low quality, draft"
)

QUALITY_SUFFIX = (
    "RAW photo, photorealistic 8k uhd, sharp focus, cinematic volumetric lighting, "
    "highly detailed textures, 35mm lens"
)

# Check PyTorch availability
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
DIFFUSERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
except ImportError:
    pass

try:
    from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler
    from PIL import Image, ImageDraw, ImageFont
    DIFFUSERS_AVAILABLE = True
except ImportError:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None


class PyTorchDiffusionEngine:
    """
    Local Diffusion Engine using PyTorch and Metal Performance Shaders (MPS).
    """

    def __init__(self, model_id: Optional[str] = None, use_mps: bool = True):
        """
        Initialize the Diffusion Engine.

        Args:
            model_id: HuggingFace model ID for diffusion (e.g., runwayml/stable-diffusion-v1-5, segmind/tiny-sd).
            use_mps: Whether to enable Apple Silicon MPS acceleration.
        """
        self.model_id = model_id or os.environ.get("DEFAULT_SD_MODEL", "segmind/tiny-sd")
        self.use_mps = use_mps and MPS_AVAILABLE
        self.device = "mps" if self.use_mps else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.initialized = False
        logger.info(f"PyTorchDiffusionEngine configured with device: {self.device}")

    def get_hardware_status(self) -> Dict[str, Any]:
        """Get details about PyTorch hardware acceleration on this Mac."""
        return {
            "torch_available": TORCH_AVAILABLE,
            "mps_available": MPS_AVAILABLE,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "active_device": self.device,
            "acceleration_label": "Apple Silicon Metal GPU (MPS)" if self.device == "mps" else self.device.upper(),
            "model_id": self.model_id,
            "initialized": self.initialized,
        }

    def initialize_pipeline(self, target_model: Optional[str] = None) -> bool:
        """Lazy initialization of diffusion model pipeline."""
        if target_model and target_model != self.model_id:
            self.model_id = target_model
            self.initialized = False
            self.pipe = None

        if self.initialized and self.pipe is not None:
            return True

        if not (TORCH_AVAILABLE and DIFFUSERS_AVAILABLE):
            logger.warning("PyTorch or Diffusers package not installed. Operating in high-speed fallback mode.")
            return False

        try:
            dtype = torch.float32  # Use float32 on MPS to prevent Metal Performance Shaders memory assertions
            logger.info(f"Loading diffusion pipeline {self.model_id} on {self.device}...")
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
            if hasattr(self.pipe, "scheduler") and hasattr(self.pipe.scheduler, "config"):
                try:
                    self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
                except Exception as sched_err:
                    logger.warning(f"Could not swap scheduler: {sched_err}")
            if hasattr(self.pipe, "enable_attention_slicing"):
                try: self.pipe.enable_attention_slicing()
                except Exception: pass
            if hasattr(self.pipe, "enable_vae_tiling"):
                try:
                    if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                        self.pipe.vae.enable_tiling()
                    else:
                        self.pipe.enable_vae_tiling()
                except Exception: pass
            self.pipe.to(self.device)
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch MPS diffusion pipeline ({self.model_id}): {e}")
            if self.model_id != "segmind/tiny-sd":
                logger.info("Falling back to local 'segmind/tiny-sd' pipeline...")
                self.model_id = "segmind/tiny-sd"
                self.initialized = False
                return self.initialize_pipeline()
            return False

    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        output_dir: str = "static/videos",
        negative_prompt: Optional[str] = None,
        raw_mode: bool = False,
        model_id: Optional[str] = None,
        quality_preset: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate image using local diffusion, or a placeholder if no model loads.

        Args:
            prompt: Text prompt for diffusion.
            width: Image width.
            height: Image height.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            output_dir: Directory to save generated PNG assets.
            negative_prompt: Optional custom negative prompt.
            raw_mode: If True, use a minimal negative prompt and never append the
                quality suffix, giving the caller direct prompt control.
            model_id: Optional model to load instead of the configured one.
            quality_preset: If True, append the photographic quality suffix to the
                prompt. Ignored when raw_mode is set.
        """
        if not prompt or not prompt.strip():
            return {"success": False, "error": "Prompt cannot be empty"}

        os.makedirs(output_dir, exist_ok=True)
        prompt_hash = hashlib.md5(f"{prompt}_{time.time()}".encode()).hexdigest()[:10]
        filename = f"diffused_{prompt_hash}.png"
        filepath = os.path.join(output_dir, filename)

        final_prompt = prompt
        if raw_mode:
            active_neg_prompt = negative_prompt or MINIMAL_NEGATIVE_PROMPT
        else:
            active_neg_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
            if quality_preset:
                final_prompt = f"{prompt}, {QUALITY_SUFFIX}"

        # Ensure final_prompt does not exceed CLIP's 77 token limit (~60 words)
        words = final_prompt.split()
        truncated = len(words) > 55
        if truncated:
            final_prompt = " ".join(words[:55])

        # Attempt hardware-accelerated generation if available
        if self.initialize_pipeline(model_id):
            try:
                if self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    try: torch.mps.empty_cache()
                    except Exception: pass

                start_time = time.time()
                steps_used = min(max(num_inference_steps, 10), MAX_INFERENCE_STEPS)
                if steps_used != num_inference_steps:
                    logger.info(
                        f"Clamped num_inference_steps {num_inference_steps} -> {steps_used} "
                        f"(allowed range 10-{MAX_INFERENCE_STEPS})"
                    )

                # Fix 3: Use CPU generator for stable seed noise initialization on MPS
                generator = None
                if TORCH_AVAILABLE:
                    generator = torch.Generator(device="cpu").manual_seed(int(time.time() * 1000) % 2147483647)

                result = self.pipe(
                    prompt=final_prompt,
                    negative_prompt=active_neg_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps_used,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                image = result.images[0]
                image.save(filepath)
                elapsed = round(time.time() - start_time, 2)

                # Convert to Base64
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                return {
                    "success": True,
                    "placeholder": False,
                    "engine": "PyTorch MPS (Apple Silicon Metal GPU)",
                    "device": self.device,
                    "model_id": self.model_id,
                    "prompt": prompt,
                    "final_prompt": final_prompt,
                    "prompt_truncated": truncated,
                    "steps_requested": num_inference_steps,
                    "steps_used": steps_used,
                    "dimensions": {"width": width, "height": height},
                    "filepath": filepath,
                    "relative_url": f"/static/videos/{filename}",
                    "base64": b64_str,
                    "inference_time_seconds": elapsed,
                }
            except Exception as e:
                logger.error(f"PyTorch MPS diffusion execution error: {e}. Retrying on CPU pipeline...")
                try:
                    self.pipe.to("cpu")
                    start_time = time.time()
                    result = self.pipe(
                        prompt=final_prompt,
                        negative_prompt=active_neg_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=12,
                        guidance_scale=guidance_scale,
                    )
                    image = result.images[0]
                    image.save(filepath)
                    elapsed = round(time.time() - start_time, 2)

                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    return {
                        "success": True,
                        "placeholder": False,
                        "engine": "PyTorch CPU Engine",
                        "device": "cpu",
                        "model_id": self.model_id,
                        "prompt": prompt,
                        "dimensions": {"width": width, "height": height},
                        "filepath": filepath,
                        "relative_url": f"/static/videos/{filename}",
                        "base64": b64_str,
                        "inference_time_seconds": elapsed,
                    }
                except Exception as cpu_err:
                    logger.error(f"PyTorch CPU fallback failed: {cpu_err}")

        # No diffusion model available: emit a clearly-labelled placeholder so the
        # caller can tell this apart from a real render.
        return self._generate_placeholder(prompt, width, height, filepath, filename)

    def _generate_placeholder(
        self, prompt: str, width: int, height: int, filepath: str, filename: str
    ) -> Dict[str, Any]:
        """
        Generate a gradient placeholder image. This is NOT a diffusion render and
        contains no image synthesis; it exists so the UI has something to show when
        the pipeline cannot load. Callers must check the "placeholder" flag.
        """
        if Image is None:
            return {
                "success": False,
                "placeholder": True,
                "engine": "Placeholder (no image backend)",
                "error": "Pillow is not installed and no diffusion pipeline loaded",
                "prompt": prompt,
                "dimensions": {"width": width, "height": height},
            }

        # Deterministic gradient derived from the prompt hash, so the same prompt
        # yields the same placeholder.
        h_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        r1, g1, b1 = (h_val & 0xFF), ((h_val >> 8) & 0xFF), ((h_val >> 16) & 0xFF)
        r2, g2, b2 = ((r1 + 100) % 255), ((g1 + 80) % 255), ((b1 + 60) % 255)

        image = Image.new("RGB", (width, height), (r1 // 2, g1 // 2, b1 // 2))
        draw = ImageDraw.Draw(image)

        for y in range(height):
            ratio = y / max(1, height)
            r = int(r1 * (1 - ratio) + r2 * ratio)
            g = int(g1 * (1 - ratio) + g2 * ratio)
            b = int(b1 * (1 - ratio) + b2 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        draw.text((25, 20), "PLACEHOLDER - no diffusion model loaded", fill=(255, 255, 255))
        draw.text((25, height - 35), f"Prompt: {prompt[:40]}...", fill=(245, 245, 245))
        image.save(filepath)

        # Convert to Base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.warning(
            "Returning gradient placeholder for prompt %r - no diffusion pipeline loaded.", prompt
        )
        return {
            "success": True,
            "placeholder": True,
            "engine": "Gradient placeholder (no diffusion model loaded)",
            "device": self.device,
            "prompt": prompt,
            "dimensions": {"width": width, "height": height},
            "filepath": filepath,
            "relative_url": f"/static/videos/{filename}",
            "base64": b64_str,
            "hardware": self.get_hardware_status(),
        }
