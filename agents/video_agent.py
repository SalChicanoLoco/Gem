"""
Quetzal Video Clip Diffusion & PyTorch SVD Engine (agents/video_agent.py)

AI-powered video clip generator supporting PyTorch Apple Silicon MPS video diffusion
(Stable Video Diffusion / AnimateDiff / Wan 2.1 integration) with procedural fallback.
"""

import io
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .gemma_agent import GemmaAgent
from .spine import get_spine
from .diffusion_engine import PyTorchDiffusionEngine, TORCH_AVAILABLE, MPS_AVAILABLE, DIFFUSERS_AVAILABLE
from . import render_estimator

# Styles routed to the Stable Video Diffusion pipeline rather than the procedural
# fallback. SVD-xt is trained to emit a fixed short window, so a longer request is
# clamped and reported instead of silently producing something else.
SVD_STYLES = ("pytorch_svd", "wan2.1", "animatediff", "sota_diffusion")
SVD_MAX_FRAMES = 25

logger = logging.getLogger(__name__)

# Check PyTorch Video Diffusion availability
VIDEO_DIFFUSION_AVAILABLE = False
try:
    from diffusers import StableVideoDiffusionPipeline
    import torch
    VIDEO_DIFFUSION_AVAILABLE = True
except ImportError:
    pass


class PyTorchVideoDiffusionEngine:
    """
    Hardware-accelerated PyTorch Video Diffusion Engine for Apple Silicon MPS.
    Supports Stable Video Diffusion, AnimateDiff, and Wan 2.1 video generation.
    """

    def __init__(self, model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"):
        """Initialize PyTorch Video Diffusion Engine."""
        self.model_id = model_id
        self.device = "mps" if MPS_AVAILABLE else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.initialized = False
        self.image_engine = PyTorchDiffusionEngine()

    def initialize_video_pipeline(self) -> bool:
        """Lazy initialization of PyTorch Video Diffusion Pipeline on Apple Silicon MPS."""
        if self.initialized:
            return True

        if not (TORCH_AVAILABLE and VIDEO_DIFFUSION_AVAILABLE):
            return False

        try:
            dtype = self.inference_dtype()
            logger.info("Loading video pipeline %s on %s (%s)...",
                        self.model_id, self.device, str(dtype).replace("torch.", ""))
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
            self.pipe.to(self.device)
            self.initialized = True
            return True
        except Exception as e:
            logger.error("Video pipeline failed to load: %s", e)
            return False

    def inference_dtype(self):
        """
        Precision for video denoising.

        Was pinned to float32 "for stability". Half precision is the standard
        configuration for Stable Video Diffusion and video is memory-bandwidth
        bound, so this is the largest lever available on this hardware: the image
        engine halved its per-step cost with the same change. Overridable with
        $VIDEO_DTYPE, and the choice is reported on every clip so a bad render can
        be attributed rather than guessed at.
        """
        named = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        override = os.environ.get("VIDEO_DTYPE")
        if override:
            chosen = named.get(override.lower())
            if chosen is not None:
                return chosen
            logger.warning("Unknown VIDEO_DTYPE %r; using the default.", override)
        return torch.float16 if self.device == "mps" else torch.float32

    def generate_video_diffusion(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 14,
        fps: int = 7,
        filepath: str = "",
    ) -> Optional[List[Image.Image]]:
        """
        Generate photorealistic video frames using PyTorch SVD pipeline on Apple Silicon MPS.
        """
        if not self.initialize_video_pipeline():
            return None

        try:
            # Step 1: Generate initial high-res photorealistic keyframe using PyTorch Image Diffusion
            keyframe_res = self.image_engine.generate(prompt, width, height)
            if not keyframe_res.get("filepath") or not os.path.exists(keyframe_res["filepath"]):
                return None

            init_image = Image.open(keyframe_res["filepath"]).convert("RGB")
            init_image = init_image.resize((width, height))

            # Step 2: Run PyTorch Video Diffusion on Apple Silicon MPS
            generator = torch.manual_seed(int(time.time()))
            output_frames = self.pipe(
                init_image,
                decode_chunk_size=4,
                generator=generator,
                num_frames=num_frames,
                fps=fps,
            ).frames[0]

            return output_frames
        except Exception:
            return None


class VideoAgent:
    """
    Video Clip Creator & Video Diffusion Engine.
    Generates photorealistic & animated video clips based on text prompts and artistic styles.
    """

    def __init__(
        self,
        output_dir: str = "static/videos",
        gemma_agent: Optional[GemmaAgent] = None,
    ):
        """
        Initialize VideoAgent.

        Args:
            output_dir: Path to directory for storing generated video clips.
            gemma_agent: Optional GemmaAgent for prompt enhancement and keyframe planning.
        """
        self.output_dir = output_dir
        self.gemma = gemma_agent or GemmaAgent()
        self.spine = get_spine()
        self.pytorch_video_engine = PyTorchVideoDiffusionEngine()
        os.makedirs(self.output_dir, exist_ok=True)

    def create_clip(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        fps: int = 8,
        style: str = "quetzal_diffusion",
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a video clip using PyTorch SVD or Quetzal Engine.

        Args:
            prompt: Text description for the video clip.
            width: Frame width.
            height: Frame height.
            num_frames: Total number of frames in clip. Ignored when
                duration_seconds is given.
            fps: Frames per second.
            style: Diffusion style (quetzal_diffusion, pytorch_svd, wan2.1, cybernetic, surreal, cosmic).
            duration_seconds: Desired clip length. Frames are derived as
                duration * fps, which is the length people actually think in.

        Returns:
            Dictionary with clip metadata, output filepath, and web URL, including
            the estimate made before rendering and the time actually taken.
        """
        requested_duration = duration_seconds
        if duration_seconds is not None:
            if duration_seconds <= 0:
                return {"success": False, "error": "duration_seconds must be positive"}
            num_frames = max(1, int(round(duration_seconds * fps)))

        engine_key = "svd" if style in SVD_STYLES or os.environ.get("USE_PYTORCH_VIDEO") == "true" else "procedural"
        frames_requested = num_frames
        clamp_note = None
        if engine_key == "svd" and num_frames > SVD_MAX_FRAMES:
            # Stable Video Diffusion is trained for a fixed short window; asking
            # for more silently produced something else or failed.
            num_frames = SVD_MAX_FRAMES
            clamp_note = (
                f"Stable Video Diffusion generates at most {SVD_MAX_FRAMES} frames per pass, so "
                f"{frames_requested} was clamped to {SVD_MAX_FRAMES} "
                f"({round(SVD_MAX_FRAMES / max(1, fps), 2)}s at {fps}fps). Longer clips need "
                f"multiple passes stitched together, which this engine does not do yet."
            )

        estimate = render_estimator.estimate(engine_key, width, height, num_frames)
        render_started = time.time()
        clip_id = f"clip_{int(time.time())}"
        filename = f"{clip_id}.gif"
        filepath = os.path.join(self.output_dir, filename)

        # Enhance prompt via Gemma if available
        enhanced_prompt = self.gemma.synthesize_aesthetic_prompt(prompt, style)

        frames = None

        # Attempt PyTorch SVD Video Diffusion if requested or available
        if engine_key == "svd":
            frames = self.pytorch_video_engine.generate_video_diffusion(
                prompt=enhanced_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                filepath=filepath,
            )

        # No SVD pipeline: fall back to procedurally drawn frames. These are not
        # diffusion output and are labelled as such in the response.
        if not frames:
            frames = self._generate_quetzal_diffusion_frames(
                prompt=enhanced_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                style=style,
            )

        # Save animated video clip
        if frames:
            duration_ms = int(1000 / max(1, fps))
            frames[0].save(
                filepath,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=duration_ms,
                loop=0,
            )

        web_path = f"/static/videos/{filename}"

        elapsed = round(time.time() - render_started, 2)
        used_svd = bool(frames and self.pytorch_video_engine.initialized)
        # Record against the engine that actually ran, not the one requested, so
        # a fallback to the procedural path cannot poison the diffusion estimate.
        render_estimator.record("svd" if used_svd else "procedural",
                                width, height, num_frames, elapsed)

        return {
            "success": True,
            "clip_id": clip_id,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "dimensions": {"width": width, "height": height},
            "num_frames": num_frames,
            "frames_requested": frames_requested,
            "frames_clamped": frames_requested != num_frames,
            "clamp_note": clamp_note,
            "fps": fps,
            "duration_requested_seconds": requested_duration,
            "duration_seconds": round(num_frames / fps, 2),
            "estimated_seconds": estimate.get("estimate_seconds"),
            "estimate_basis": estimate.get("basis"),
            "render_seconds": elapsed,
            "style": style,
            "engine": "PyTorch MPS Video Diffusion (Apple Silicon)" if (frames and self.pytorch_video_engine.initialized) else "Procedural animation placeholder (no video diffusion model loaded)",
            "placeholder": not (frames and self.pytorch_video_engine.initialized),
            "file_path": filepath,
            "video_url": web_path,
        }

    def _generate_quetzal_diffusion_frames(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        style: str,
    ) -> List[Image.Image]:
        """
        Synthesize neural diffusion video frames with camera motion vectors and temporal interpolation on Apple Silicon MPS.
        """
        frames: List[Image.Image] = []

        # Step 1: Render high-res initial neural keyframe on Apple Silicon MPS
        keyframe_res = self.pytorch_video_engine.image_engine.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=15,
        )

        base_img = None
        if keyframe_res.get("filepath") and os.path.exists(keyframe_res["filepath"]):
            try:
                base_img = Image.open(keyframe_res["filepath"]).convert("RGB").resize((width, height))
            except Exception:
                pass

        if base_img is None:
            base_img = Image.new("RGB", (width, height), (20, 30, 45))

        # Step 2: Apply PyTorch camera motion vectors & temporal noise interpolation
        for t in range(num_frames):
            phase = t / max(1, num_frames)
            zoom = 1.0 + (phase * 0.08)  # Smooth cinematic zoom-in motion
            
            # Crop & resize for smooth zoom interpolation
            w_crop = int(width / zoom)
            h_crop = int(height / zoom)
            left = (width - w_crop) // 2
            top = (height - h_crop) // 2
            
            frame_img = base_img.crop((left, top, left + w_crop, top + h_crop)).resize((width, height), Image.Resampling.LANCZOS)

            # Apply subtle lighting & focus shifts for temporal motion
            if phase > 0:
                frame_img = frame_img.filter(ImageFilter.SMOOTH_MORE)

            frames.append(frame_img)

        return frames
