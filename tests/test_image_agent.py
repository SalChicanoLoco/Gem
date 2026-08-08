"""
Tests for the ImageAgent class.
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.image_agent import ImageAgent


@pytest.fixture
def agent():
    """Create an ImageAgent instance for testing."""
    ag = ImageAgent()
    return ag


class TestImageAgentInit:
    """Tests for ImageAgent initialization."""

    def test_init_without_credentials(self):
        """Test initialization without API credentials."""
        agent = ImageAgent()
        assert agent.api_key is None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        agent = ImageAgent(api_key="test_key")
        assert agent.api_key == "test_key"

    def test_init_with_custom_endpoint(self):
        """Test initialization with custom endpoint."""
        agent = ImageAgent(api_endpoint="https://custom.api.com")
        assert agent.api_endpoint == "https://custom.api.com"


class TestImageGeneration:
    """Tests for image generation."""

    def test_generate_with_valid_prompt(self, agent):
        """Test generation with valid prompt returns response."""
        with patch.object(agent.diffusion_engine, "generate") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "engine": "PyTorch MPS (Apple Silicon Metal GPU)",
                "dimensions": {"width": 512, "height": 512},
                "filepath": "static/videos/test.png",
                "relative_url": "/static/videos/test.png",
                "base64": "mock_b64",
            }
            result = agent.generate_image("A beautiful sunset")
            assert result["success"] is True

    def test_generate_with_empty_prompt(self, agent):
        """Test generation with empty prompt returns error."""
        result = agent.generate_image("")
        assert result["success"] is False
        assert "error" in result

    def test_generate_with_dimensions(self, agent):
        """Test generation with custom dimensions."""
        with patch.object(agent.diffusion_engine, "generate") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "engine": "PyTorch MPS (Apple Silicon Metal GPU)",
                "dimensions": {"width": 1024, "height": 768},
                "filepath": "static/videos/test.png",
                "relative_url": "/static/videos/test.png",
                "base64": "mock_b64",
            }
            result = agent.generate_image("Test image", width=1024, height=768)
            assert result["success"] is True
            assert result["dimensions"]["width"] == 1024
            assert result["dimensions"]["height"] == 768

    def test_generate_with_invalid_dimensions(self, agent):
        """Test generation with invalid dimensions returns error."""
        result = agent.generate_image("Test image", width=10, height=100)
        assert result["success"] is False
        assert "Dimensions" in result["error"]

    def test_generate_with_style(self, agent):
        """Test generation with style parameter."""
        with patch.object(agent.diffusion_engine, "generate") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "engine": "PyTorch MPS (Apple Silicon Metal GPU)",
                "style": "watercolor",
                "dimensions": {"width": 512, "height": 512},
                "filepath": "static/videos/test.png",
                "relative_url": "/static/videos/test.png",
                "base64": "mock_b64",
            }
            result = agent.generate_image("Test image", style="watercolor")
            assert result["success"] is True
            assert result["style"] == "watercolor"


class TestPromptValidation:
    """Tests for prompt validation."""

    def test_validate_empty_prompt(self, agent):
        """Test validation of empty prompt."""
        result = agent.validate_prompt("")
        assert result["valid"] is False

    def test_validate_short_prompt(self, agent):
        """Test validation of prompt that's too short."""
        result = agent.validate_prompt("ab")
        assert result["valid"] is False

    def test_validate_long_prompt(self, agent):
        """Test validation of prompt that's too long."""
        result = agent.validate_prompt("x" * 1001)
        assert result["valid"] is False

    def test_validate_valid_prompt(self, agent):
        """Test validation of valid prompt."""
        result = agent.validate_prompt("A serene mountain landscape")
        assert result["valid"] is True
        assert "processed_prompt" in result
        assert "character_count" in result

    def test_validate_strips_whitespace(self, agent):
        """Test that validation strips whitespace."""
        result = agent.validate_prompt("  test prompt  ")
        assert result["valid"] is True
        assert result["processed_prompt"] == "test prompt"


class TestStylesList:
    """Tests for available styles."""

    def test_list_styles_returns_list(self, agent):
        """Test that list_styles returns a list."""
        styles = agent.list_styles()
        assert isinstance(styles, list)
        assert len(styles) > 0

    def test_list_styles_contains_common_styles(self, agent):
        """Test that list includes common art styles."""
        styles = agent.list_styles()
        assert "realistic" in styles
        assert "artistic" in styles
        assert "cartoon" in styles


class TestImageStatus:
    """Tests for image status checking."""

    def test_status_mock_image(self, agent):
        """Test status check for mock image."""
        result = agent.get_image_status("mock_abc123")
        assert result["success"] is True
        assert result["status"] == "completed"

    def test_status_empty_id(self, agent):
        """Test status check with empty ID."""
        result = agent.get_image_status("")
        assert result["success"] is False

    def test_status_without_api_key(self, agent):
        """Test status check without API key for real ID."""
        result = agent.get_image_status("real_image_id")
        assert result["success"] is False
        assert "API key" in result["error"]
