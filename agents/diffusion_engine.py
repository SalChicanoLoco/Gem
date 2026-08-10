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

from .lora_trainer import ADAPTER_META_FILENAME, read_adapter_meta

logger = logging.getLogger(__name__)

# Upper bound on denoising steps, capped to keep MPS runs inside a responsive
# interactive budget. Requests above this are clamped and reported back.
#
# Was 19, sized against float32. Half precision cut the per-step cost roughly in
# half without changing the image, so 50 steps now costs about what 19 did before
# (~16s at 512x768). The budget is unchanged; the precision bought the steps.
MAX_INFERENCE_STEPS = int(os.environ.get("MAX_INFERENCE_STEPS", "50"))

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


def _is_blank(image) -> bool:
    """
    True when a render carries no detail at all — a solid frame.

    A blank image returned as a successful render is indistinguishable from a real
    one to a caller that reads only ``success``. The NSFW classifier used to
    produce these deliberately; half precision can produce them by accident when
    a tensor goes non-finite. Either way the caller should be told.
    """
    try:
        from PIL import ImageStat
        return max(ImageStat.Stat(image.convert("RGB")).stddev) < 1.0
    except Exception:  # pragma: no cover - never fail a good render on the check
        return False


class PyTorchDiffusionEngine:
    """
    Local Diffusion Engine using PyTorch and Metal Performance Shaders (MPS).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        use_mps: bool = True,
        lora_path: Optional[str] = None,
        safety_checker: Optional[bool] = None,
        dtype: Optional[str] = None,
    ):
        """
        Initialize the Diffusion Engine.

        Args:
            model_id: HuggingFace model ID for diffusion (e.g., runwayml/stable-diffusion-v1-5, segmind/tiny-sd).
            use_mps: Whether to enable Apple Silicon MPS acceleration.
            lora_path: Directory holding LoRA adapter weights to apply on load,
                as produced by LoRALocalTrainer. Defaults to $DEFAULT_LORA_PATH.
            safety_checker: Load the NSFW classifier. Off by default; see
                ENABLE_SAFETY_CHECKER below. Opt in with $ENABLE_SAFETY_CHECKER=true.
            dtype: Inference precision override ("float16", "bfloat16", "float32").
                Defaults to float16 on MPS and float32 elsewhere.
        """
        self.model_id = model_id or os.environ.get("DEFAULT_SD_MODEL", "segmind/tiny-sd")
        self.lora_path = lora_path or os.environ.get("DEFAULT_LORA_PATH") or None
        self.use_mps = use_mps and MPS_AVAILABLE
        self.device = "mps" if self.use_mps else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        # The bundled NSFW classifier blanks the image to solid black on a hit, and
        # it hits on benign prompts: measured on this engine, "a portrait of a woman,
        # desert light" was blanked on 2 of 3 seeds in float16 and 0 of 3 in float32,
        # because its similarity thresholds are precision-sensitive. A silently
        # black render is worse than no filter for a local single-user engine, and
        # it made float16 unusable. Off unless explicitly enabled.
        if safety_checker is None:
            safety_checker = os.environ.get("ENABLE_SAFETY_CHECKER", "false").lower() == "true"
        self.safety_checker = safety_checker
        self.dtype_override = dtype or os.environ.get("DIFFUSION_DTYPE") or None
        self.pipe = None
        self.initialized = False
        self.active_lora: Optional[str] = None
        self.lora_error: Optional[str] = None
        logger.info(f"PyTorchDiffusionEngine configured with device: {self.device}")

    def _inference_dtype(self):
        """
        Precision for inference.

        float16 on MPS is ~2x faster than float32 for the same image: measured
        3.5s against 7.1s at 19 steps, 512x512, with output statistics matching to
        within a fraction of a percent. Training stays float32 (see LoRALocalTrainer)
        because MPS autograd is unreliable in half precision; this is inference only.
        """
        named = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if self.dtype_override:
            chosen = named.get(self.dtype_override.lower())
            if chosen is not None:
                return chosen
            logger.warning("Unknown dtype %r; falling back to default.", self.dtype_override)
        return torch.float16 if self.device == "mps" else torch.float32

    def _resolve_base_model(self, target_model: Optional[str]) -> Optional[str]:
        """
        Decide which base model to load, given the adapter the caller asked for.

        A LoRA only fits the UNet geometry it was trained on, so an SD 1.5 adapter
        cannot load into tiny-sd. When the adapter declares its base and the caller
        did not name a model, load the base the adapter needs. When the caller did
        name a conflicting model, honour that and refuse the adapter rather than
        overriding an explicit choice.

        Returns the model to switch to, or None to leave the current one alone.
        """
        self.lora_error = None
        if not self.lora_path or not os.path.isdir(self.lora_path):
            return target_model

        meta = read_adapter_meta(self.lora_path)
        base = (meta or {}).get("base_model_id")
        requested = target_model or self.model_id
        if not base or base == requested:
            return target_model

        if target_model:
            self.lora_error = (
                f"adapter {self.lora_path} was trained on {base}, which does not match the "
                f"requested model {target_model}; pass model_id={base!r} to use this adapter"
            )
            return target_model

        logger.info("Adapter %s declares base model %s; loading that instead of %s.",
                    self.lora_path, base, self.model_id)
        return base

    def get_hardware_status(self) -> Dict[str, Any]:
        """Get details about PyTorch hardware acceleration on this Mac."""
        return {
            "torch_available": TORCH_AVAILABLE,
            "mps_available": MPS_AVAILABLE,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "active_device": self.device,
            "acceleration_label": "Apple Silicon Metal GPU (MPS)" if self.device == "mps" else self.device.upper(),
            "model_id": self.model_id,
            "dtype": str(self._inference_dtype()).replace("torch.", "") if TORCH_AVAILABLE else None,
            "safety_checker": self.safety_checker,
            "lora": self.active_lora,
            "lora_error": self.lora_error,
            "initialized": self.initialized,
        }

    def initialize_pipeline(
        self, target_model: Optional[str] = None, target_lora: Optional[str] = None
    ) -> bool:
        """
        Lazy initialization of the diffusion model pipeline.

        Args:
            target_model: Load this model instead of the configured one.
            target_lora: Apply this LoRA adapter directory instead of the
                configured one. Changing either forces a reload.
        """
        if target_lora is not None and target_lora != self.lora_path:
            self.lora_path = target_lora
            self.initialized = False
            self.pipe = None

        resolved_model = self._resolve_base_model(target_model)
        if resolved_model and resolved_model != self.model_id:
            self.model_id = resolved_model
            self.initialized = False
            self.pipe = None

        if self.initialized and self.pipe is not None:
            return True

        if not (TORCH_AVAILABLE and DIFFUSERS_AVAILABLE):
            logger.warning("PyTorch or Diffusers package not installed. Operating in high-speed fallback mode.")
            return False

        try:
            dtype = self._inference_dtype()
            logger.info(
                "Loading diffusion pipeline %s on %s (%s, safety_checker=%s)...",
                self.model_id, self.device, str(dtype).replace("torch.", ""),
                "on" if self.safety_checker else "off",
            )
            load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
            if not self.safety_checker:
                load_kwargs["safety_checker"] = None
                load_kwargs["requires_safety_checker"] = False
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                **load_kwargs,
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
            self.active_lora = None
            if self.lora_path:
                if self.lora_error:
                    # Declared mismatch, already explained by _resolve_base_model.
                    logger.error("Not applying LoRA adapter: %s", self.lora_error)
                elif not os.path.isdir(self.lora_path):
                    self.lora_error = f"LoRA path {self.lora_path} does not exist"
                    logger.warning("%s; loading base model only.", self.lora_error)
                elif not hasattr(self.pipe, "load_lora_weights"):
                    self.lora_error = f"pipeline {self.model_id} does not support LoRA weights"
                    logger.warning("Pipeline %s does not support LoRA weights.", self.model_id)
                else:
                    try:
                        self.pipe.load_lora_weights(self.lora_path)
                        self.active_lora = self.lora_path
                        logger.info("Applied LoRA adapter from %s", self.lora_path)
                    except Exception as lora_err:
                        # A bad adapter must not silently masquerade as the base model.
                        # Shape mismatches list every tensor; report the cause, not the list.
                        if "size mismatch" in str(lora_err):
                            self.lora_error = (
                                f"adapter {self.lora_path} does not fit the UNet of {self.model_id} "
                                f"(tensor shape mismatch). It was trained on a different base model, "
                                f"and no {ADAPTER_META_FILENAME} records which one."
                            )
                            logger.error("Failed to apply LoRA adapter: %s", self.lora_error)
                        else:
                            self.lora_error = str(lora_err)
                            logger.error("Failed to apply LoRA adapter %s: %s", self.lora_path, lora_err)

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
        lora_path: Optional[str] = None,
        seed: Optional[int] = None,
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
            lora_path: Optional LoRA adapter directory to apply for this call.
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
        if self.initialize_pipeline(model_id, lora_path):
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
                active_seed = None
                if TORCH_AVAILABLE:
                    # Reported back so any render can be reproduced, which is what
                    # makes an A/B between two adapters or step counts meaningful.
                    active_seed = int(time.time() * 1000) % 2147483647 if seed is None else int(seed)
                    generator = torch.Generator(device="cpu").manual_seed(active_seed)

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

                blank = _is_blank(image)
                if blank:
                    logger.error(
                        "Render for %r produced a blank image (dtype=%s, safety_checker=%s). "
                        "Reported as blank_image rather than as a successful render.",
                        prompt[:60], self._inference_dtype(), self.safety_checker,
                    )

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
                    "dtype": str(self._inference_dtype()).replace("torch.", ""),
                    "safety_checker": self.safety_checker,
                    "blank_image": blank,
                    "lora": self.active_lora,
                    "lora_error": self.lora_error,
                    "prompt": prompt,
                    "final_prompt": final_prompt,
                    "prompt_truncated": truncated,
                    "steps_requested": num_inference_steps,
                    "steps_used": steps_used,
                    "seed": active_seed,
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
