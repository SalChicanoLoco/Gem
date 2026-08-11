"""
ImageAgent - Image generation agent for creating visualizations and AI-generated images.
"""

import hashlib
import os
from typing import Any
import requests
from .diffusion_engine import PyTorchDiffusionEngine


class ImageAgent:
    """
    Agent responsible for image generation tasks.
    Integrates external image APIs and local PyTorch MPS hardware-accelerated diffusion.
    """

    def __init__(self, api_key: str | None = None, api_endpoint: str | None = None):
        """
        Initialize the ImageAgent.

        Args:
            api_key: API key for external image generation service.
            api_endpoint: Base URL for the image generation API.
        """
        self.api_key = api_key or os.environ.get("IMAGE_API_KEY")
        self.api_endpoint = api_endpoint or os.environ.get(
            "IMAGE_API_ENDPOINT", "https://api.example.com/v1/images"
        )
        self.diffusion_engine = PyTorchDiffusionEngine()

    def generate_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        style: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate an image based on a text prompt.

        Args:
            prompt: Text description of the image to generate.
            width: Image width in pixels.
            height: Image height in pixels.
            style: Optional style modifier (e.g., "realistic", "artistic", "cartoon").

        Returns:
            Dictionary with generation results and image data/URL.
        """
        if not prompt or not prompt.strip():
            return {"success": False, "error": "Prompt cannot be empty"}

        # Validate dimensions
        if width < 64 or width > 2048 or height < 64 or height > 2048:
            return {
                "success": False,
                "error": "Dimensions must be between 64 and 2048 pixels",
            }

        # Check if local PyTorch diffusion is requested
        if style in ["local_mps", "diffusion", "pytorch"] or os.environ.get("USE_LOCAL_DIFFUSION") == "true":
            return self.diffusion_engine.generate(prompt, width, height)

        # If API key is configured, attempt real generation
        if self.api_key:
            return self._call_external_api(prompt, width, height, style)

        # Fall back to local PyTorch diffusion / procedural synthesis engine
        res = self.diffusion_engine.generate(prompt, width, height)
        res["style"] = style
        return res

    def _call_external_api(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str | None,
    ) -> dict[str, Any]:
        """
        Call external image generation API.

        Returns:
            Dictionary with API response or error.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }

        if style:
            payload["style"] = style

        try:
            response = requests.post(
                f"{self.api_endpoint}/generate",
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            return {
                "success": True,
                "image_url": data.get("url"),
                "image_id": data.get("id"),
                "prompt": prompt,
                "dimensions": {"width": width, "height": height},
            }

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"API request failed: {str(e)}"}

    def get_image_status(self, image_id: str) -> dict[str, Any]:
        """
        Check the status of an image generation request.

        Args:
            image_id: The ID of the image generation request.

        Returns:
            Dictionary with status information.
        """
        if not image_id:
            return {"success": False, "error": "Image ID is required"}

        if image_id.startswith("mock_"):
            return {
                "success": True,
                "image_id": image_id,
                "status": "completed",
                "mock": True,
            }

        if not self.api_key:
            return {"success": False, "error": "API key not configured"}

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.api_endpoint}/status/{image_id}",
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return {"success": True, **response.json()}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Status check failed: {str(e)}"}

    def list_styles(self) -> list[str]:
        """
        List available image generation styles.

        Returns:
            List of available style names.
        """
        return [
            "realistic",
            "artistic",
            "cartoon",
            "watercolor",
            "oil-painting",
            "sketch",
            "digital-art",
            "3d-render",
        ]

    def validate_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Validate and preprocess a prompt for image generation.

        Args:
            prompt: The prompt to validate.

        Returns:
            Dictionary with validation results.
        """
        if not prompt:
            return {"valid": False, "error": "Prompt is required"}

        prompt = prompt.strip()

        if len(prompt) < 3:
            return {"valid": False, "error": "Prompt must be at least 3 characters"}

        if len(prompt) > 1000:
            return {"valid": False, "error": "Prompt must be less than 1000 characters"}

        return {
            "valid": True,
            "processed_prompt": prompt,
            "character_count": len(prompt),
        }
