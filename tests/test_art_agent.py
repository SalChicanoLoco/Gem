"""
Tests for the ArtAgent class.
"""

import pytest
from agents.art_agent import ArtAgent


@pytest.fixture
def agent():
    """Create an ArtAgent instance for testing."""
    return ArtAgent()


class TestArtAgentInit:
    """Tests for ArtAgent initialization."""

    def test_init_without_credentials(self):
        """Test initialization without API credentials."""
        agent = ArtAgent()
        assert agent.api_key is None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        agent = ArtAgent(api_key="test_key")
        assert agent.api_key == "test_key"

    def test_init_with_custom_endpoint(self):
        """Test initialization with custom endpoint."""
        agent = ArtAgent(vision_endpoint="https://custom.api.com")
        assert agent.vision_endpoint == "https://custom.api.com"


class TestAestheticAnalysis:
    """Tests for aesthetic analysis."""

    def test_analyze_refuses_rather_than_scoring_an_unseen_image(self, agent):
        """
        There is no vision model in this build. Scores were previously derived
        from an MD5 of the image URL, so a URL that did not exist still received
        composition, colour-harmony and lighting numbers.
        """
        result = agent.analyze_aesthetics({"image_url": "https://example.com/image.jpg"})
        assert result["success"] is False
        assert result["analysed"] is False
        assert "not implemented" in result["error"]

    def test_analyze_returns_no_scores_at_all(self, agent):
        """A refusal must not carry score-shaped fields a caller could read."""
        result = agent.analyze_aesthetics({"image_base64": "base64encodeddata"})
        for key in ("aesthetic_scores", "style_classification", "mood", "dominant_colors"):
            assert key not in result

    def test_analyze_without_image(self, agent):
        """Test analysis without image data returns error."""
        result = agent.analyze_aesthetics({})
        assert result["success"] is False
        assert "error" in result



class TestImageSets:
    """Tests for image set management."""

    def test_create_image_set(self, agent):
        """Test creating an image set."""
        result = agent.create_image_set(
            name="Test Set",
            description="A test image set",
            base_style="artistic",
            target_count=5,
        )
        assert result["success"] is True
        assert "image_set" in result
        assert result["image_set"]["name"] == "Test Set"

    def test_create_image_set_without_name(self, agent):
        """Test creating image set without name returns error."""
        result = agent.create_image_set(name="", description="Test")
        assert result["success"] is False

    def test_add_to_image_set(self, agent):
        """Test adding image to a set."""
        # Create set first
        create_result = agent.create_image_set(
            name="Test Set",
            description="Test",
            target_count=2,
        )
        set_id = create_result["image_set"]["id"]

        # Add image
        add_result = agent.add_to_image_set(
            set_id=set_id,
            image_data={"image_url": "https://example.com/img1.jpg"},
        )
        assert add_result["success"] is True
        assert add_result["current_count"] == 1

    def test_image_set_completion(self, agent):
        """Test that image set status changes when full."""
        create_result = agent.create_image_set(
            name="Small Set",
            description="Test",
            target_count=1,
        )
        set_id = create_result["image_set"]["id"]

        add_result = agent.add_to_image_set(
            set_id=set_id,
            image_data={"image_url": "https://example.com/img.jpg"},
        )
        assert add_result["set_status"] == "complete"

    def test_add_to_nonexistent_set(self, agent):
        """Test adding to nonexistent set returns error."""
        result = agent.add_to_image_set(
            set_id="nonexistent",
            image_data={"image_url": "https://example.com/img.jpg"},
        )
        assert result["success"] is False

    def test_get_image_set(self, agent):
        """Test retrieving an image set."""
        create_result = agent.create_image_set(
            name="Retrieve Test",
            description="Test",
        )
        set_id = create_result["image_set"]["id"]

        get_result = agent.get_image_set(set_id)
        assert get_result["success"] is True
        assert get_result["image_set"]["name"] == "Retrieve Test"

    def test_get_nonexistent_set(self, agent):
        """Test getting nonexistent set returns error."""
        result = agent.get_image_set("nonexistent")
        assert result["success"] is False


class TestContextExport:
    """Tests for context export functionality."""

    def test_export_context(self, agent):
        """Test exporting context from image set."""
        create_result = agent.create_image_set(
            name="Context Test",
            description="For context testing",
            base_style="minimalist",
        )
        set_id = create_result["image_set"]["id"]

        export_result = agent.export_context(set_id)
        assert export_result["success"] is True
        assert "context" in export_result
        assert export_result["context"]["style"] == "minimalist"

    def test_export_nonexistent_set(self, agent):
        """Test exporting from nonexistent set returns error."""
        result = agent.export_context("nonexistent")
        assert result["success"] is False


class TestStyleTransfer:
    """Tests for style transfer functionality."""

    def test_apply_style_transfer(self, agent):
        """Test applying style transfer."""
        result = agent.apply_style_transfer(
            source_image={"image_url": "https://example.com/source.jpg"},
            target_style="impressionist",
            intensity=0.8,
        )
        assert result["success"] is False
        assert "not implemented" in result["error"]
        assert result["target_style"] == "impressionist"

    def test_style_transfer_without_image(self, agent):
        """Test style transfer without image returns error."""
        result = agent.apply_style_transfer(
            source_image={},
            target_style="artistic",
        )
        assert result["success"] is False

    def test_style_transfer_without_style(self, agent):
        """Test style transfer without target style returns error."""
        result = agent.apply_style_transfer(
            source_image={"image_url": "https://example.com/img.jpg"},
            target_style="",
        )
        assert result["success"] is False

    def test_intensity_clamping(self, agent):
        """Test that intensity is clamped to valid range."""
        result = agent.apply_style_transfer(
            source_image={"image_url": "https://example.com/img.jpg"},
            target_style="artistic",
            intensity=1.5,  # Above max
        )
        assert result["intensity"] == 1.0


class TestAvailableStyles:
    """Tests for available styles."""

    def test_list_styles(self, agent):
        """Test listing available styles."""
        styles = agent.list_available_styles()
        assert isinstance(styles, list)
        assert len(styles) > 0
        assert "realistic" in styles
        assert "artistic" in styles
