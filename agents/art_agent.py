"""
ArtAgent - Agent for aesthetics, computer vision, and scalable image set generation.
"""

import hashlib
import os
import uuid
from typing import Any


from .gemma_agent import GemmaAgent


class ArtAgent:
    """
    Agent responsible for aesthetic analysis, computer vision tasks, and
    generating scalable sets of images that can work with other agents for context.
    
    Integrated with local Gemma LLM for multi-modal aesthetic prompt synthesis.
    """

    def __init__(
        self,
        api_key: str | None = None,
        vision_endpoint: str | None = None,
        gemma_agent: GemmaAgent | None = None,
    ):
        """
        Initialize the ArtAgent.

        Args:
            api_key: API key for external vision/aesthetic services.
            vision_endpoint: Base URL for vision API.
            gemma_agent: Optional GemmaAgent for LLM aesthetic prompt synthesis.
        """
        self.api_key = api_key or os.environ.get("VISION_API_KEY")
        self.vision_endpoint = vision_endpoint or os.environ.get(
            "VISION_API_ENDPOINT", "https://api.example.com/v1/vision"
        )
        self.gemma = gemma_agent or GemmaAgent()
        self._image_sets: dict[str, dict[str, Any]] = {}

    def analyze_aesthetics(self, image_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze the aesthetic qualities of an image.

        Args:
            image_data: Dictionary containing image URL or base64 data.
                - image_url: str (URL to image)
                - image_base64: str (base64-encoded image data)

        Returns:
            Dictionary with aesthetic analysis results.
        """
        if not image_data.get("image_url") and not image_data.get("image_base64"):
            return {"success": False, "error": "image_url or image_base64 is required"}

        # There is no vision backend, so there is no analysis.
        #
        # This used to derive "aesthetic_scores" from an MD5 of the image URL and
        # return them with mock: true. The numbers moved when the URL changed, so
        # they looked responsive, but nothing ever opened the image: a URL that
        # does not exist scored 48.63 for composition, colour harmony and
        # lighting. A refusal is more useful than a plausible number.
        #
        # No client is written here for a backend that does not exist yet, since
        # an untested integration against the placeholder api.example.com endpoint
        # would be the next thing to quietly stop meaning anything. For measurable
        # image properties on local files, agents/image_metrics.py does real work.
        return {
            "success": False,
            "analysed": False,
            "error": (
                "aesthetic analysis is not implemented: this build has no vision model. "
                "agents/image_metrics.py provides measured SSIM and colour-histogram "
                "comparison for local images."
            ),
        }

    def create_image_set(
        self,
        name: str,
        description: str,
        base_style: str | None = None,
        target_count: int = 10,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new scalable image set configuration.

        Args:
            name: Name of the image set.
            description: Description of the set's purpose.
            base_style: Base artistic style for the set.
            target_count: Target number of images in the set.
            context: Additional context from other agents.

        Returns:
            Dictionary with the created image set details.
        """
        if not name or not name.strip():
            return {"success": False, "error": "Name is required"}

        set_id = str(uuid.uuid4())[:8]

        image_set = {
            "id": set_id,
            "name": name.strip(),
            "description": description,
            "base_style": base_style or "realistic",
            "target_count": max(1, min(target_count, 100)),
            "current_count": 0,
            "images": [],
            "context": context or {},
            "status": "created",
            "aesthetic_profile": self._generate_aesthetic_profile(base_style),
        }

        self._image_sets[set_id] = image_set

        return {
            "success": True,
            "image_set": {
                "id": set_id,
                "name": image_set["name"],
                "status": "created",
                "target_count": image_set["target_count"],
            },
        }

    def _generate_aesthetic_profile(self, style: str | None) -> dict[str, Any]:
        """Generate an aesthetic profile for consistent image generation."""
        profiles = {
            "realistic": {
                "saturation": 0.7, "contrast": 0.6, "brightness": 0.5,
                "detail_level": "high", "color_temperature": "neutral",
            },
            "artistic": {
                "saturation": 0.8, "contrast": 0.7, "brightness": 0.6,
                "detail_level": "medium", "color_temperature": "warm",
            },
            "minimalist": {
                "saturation": 0.3, "contrast": 0.4, "brightness": 0.7,
                "detail_level": "low", "color_temperature": "cool",
            },
            "surrealist": {
                "saturation": 0.9, "contrast": 0.8, "brightness": 0.5,
                "detail_level": "high", "color_temperature": "varied",
            },
        }
        return profiles.get(style or "realistic", profiles["realistic"])

    def add_to_image_set(
        self,
        set_id: str,
        image_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Add an image to an existing image set.

        Args:
            set_id: ID of the image set.
            image_data: Image data including URL or generation params.

        Returns:
            Dictionary with the result of adding the image.
        """
        if set_id not in self._image_sets:
            return {"success": False, "error": "Image set not found"}

        image_set = self._image_sets[set_id]

        if image_set["current_count"] >= image_set["target_count"]:
            return {"success": False, "error": "Image set is at capacity"}

        # Analyze aesthetics if image URL provided
        if image_data.get("image_url"):
            aesthetic_result = self.analyze_aesthetics(image_data)
            image_data["aesthetic_analysis"] = aesthetic_result.get("aesthetic_scores")

        image_id = str(uuid.uuid4())[:8]
        image_entry = {
            "id": image_id,
            "data": image_data,
            "position": image_set["current_count"],
        }

        image_set["images"].append(image_entry)
        image_set["current_count"] += 1

        if image_set["current_count"] >= image_set["target_count"]:
            image_set["status"] = "complete"
        else:
            image_set["status"] = "in_progress"

        return {
            "success": True,
            "image_id": image_id,
            "set_status": image_set["status"],
            "current_count": image_set["current_count"],
            "target_count": image_set["target_count"],
        }

    def get_image_set(self, set_id: str) -> dict[str, Any]:
        """
        Retrieve an image set by ID.

        Args:
            set_id: ID of the image set.

        Returns:
            Dictionary with image set details.
        """
        if set_id not in self._image_sets:
            return {"success": False, "error": "Image set not found"}

        return {"success": True, "image_set": self._image_sets[set_id]}

    def export_context(self, set_id: str) -> dict[str, Any]:
        """
        Export context from an image set for use by other agents.

        Args:
            set_id: ID of the image set.

        Returns:
            Dictionary with exportable context for agent coordination.
        """
        if set_id not in self._image_sets:
            return {"success": False, "error": "Image set not found"}

        image_set = self._image_sets[set_id]

        # Aggregate aesthetic data for context
        aesthetic_summary = self._aggregate_aesthetics(image_set["images"])

        return {
            "success": True,
            "context": {
                "set_id": set_id,
                "name": image_set["name"],
                "description": image_set["description"],
                "style": image_set["base_style"],
                "aesthetic_profile": image_set["aesthetic_profile"],
                "aesthetic_summary": aesthetic_summary,
                "image_count": image_set["current_count"],
                "status": image_set["status"],
            },
        }

    def _aggregate_aesthetics(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate aesthetic scores across all images in a set."""
        if not images:
            return {}

        scores_sum: dict[str, float] = {}
        count = 0

        for img in images:
            analysis = img.get("data", {}).get("aesthetic_analysis", {})
            if analysis:
                count += 1
                for key, value in analysis.items():
                    if isinstance(value, (int, float)):
                        scores_sum[key] = scores_sum.get(key, 0) + value

        if count == 0:
            return {}

        return {k: round(v / count, 2) for k, v in scores_sum.items()}

    def apply_style_transfer(
        self,
        source_image: dict[str, Any],
        target_style: str,
        intensity: float = 0.7,
    ) -> dict[str, Any]:
        """
        Apply style transfer to an image (mock implementation).

        Args:
            source_image: Source image data.
            target_style: Target artistic style.
            intensity: Style transfer intensity (0.0 to 1.0).

        Returns:
            Dictionary with style transfer results.
        """
        if not source_image.get("image_url") and not source_image.get("image_base64"):
            return {"success": False, "error": "Source image is required"}

        if not target_style:
            return {"success": False, "error": "Target style is required"}

        intensity = max(0.0, min(1.0, intensity))

        # Returned success: True with a fresh uuid as a "result_id", so a caller
        # reading only success believed an image had been produced. No transfer
        # happens here and none is claimed.
        return {
            "success": False,
            "original_image": source_image,
            "target_style": target_style,
            "intensity": intensity,
            "error": (
                "style transfer is not implemented: this build has no style-transfer model. "
                "For local stylisation, train a LoRA with agents/lora_trainer.py and apply it "
                "via the diffusion engine's lora_path."
            ),
        }

    def list_available_styles(self) -> list[str]:
        """List available artistic styles for image sets."""
        return [
            "realistic",
            "artistic",
            "minimalist",
            "surrealist",
            "impressionist",
            "abstract",
            "pop-art",
            "watercolor",
            "sketch",
            "digital-art",
            "3d-render",
            "anime",
            "photographic",
            "cinematic",
        ]
