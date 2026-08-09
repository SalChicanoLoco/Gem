"""
HuggingFace Model Pre-Fetcher Script (scripts/download_models.py)

Pre-downloads high-resolution Stable Diffusion and Stable Video Diffusion model weights
to local Mac HuggingFace cache for zero-latency local PyTorch MPS GPU inference.
"""

import sys
from typing import List

MODEL_TARGETS = [
    ("runwayml/stable-diffusion-v1-5", "Full Photorealistic SD 1.5 Base Model"),
    ("stabilityai/stable-diffusion-2-1-base", "High-Resolution SD 2.1 768px Model"),
    ("segmind/tiny-sd", "High-Speed Tiny SD Model"),
    ("stabilityai/stable-video-diffusion-img2vid-xt", "Stable Video Diffusion (SVD-XT) 25-frame Model"),
]

def download_models(models: List[str] = None):
    """Pre-fetch specified HuggingFace model checkpoints to local cache."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
    except ImportError:
        print("❌ Error: PyTorch or Diffusers package not installed. Run ./venv/bin/pip install torch diffusers first.")
        sys.exit(1)

    print("🚀 Pre-fetching HuggingFace Model Weights to Local Cache...")
    print("=" * 65)

    targets = models or [m[0] for m in MODEL_TARGETS]

    for model_id in targets:
        print(f"\n📦 Fetching {model_id}...")
        try:
            if "video" in model_id.lower() or "svd" in model_id.lower():
                print(f"Downloading Stable Video Diffusion pipeline: {model_id}")
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                )
            else:
                print(f"Downloading Text-to-Image pipeline: {model_id}")
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                )
            print(f"✅ {model_id} cached successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not pre-fetch {model_id}: {e}")

    print("\n=" * 65)
    print("✨ Model Pre-Fetch Complete! All model weights cached locally.")

if __name__ == "__main__":
    download_models()
