"""
Tests for the Flask API endpoints.
"""

import pytest
from api.app import create_app


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Test that health endpoint returns JSON."""
        response = client.get("/")
        assert response.content_type == "application/json"

    def test_health_contains_status(self, client):
        """Test that health response contains status field."""
        response = client.get("/")
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_service_info(self, client):
        """Test that health response contains service information."""
        response = client.get("/")
        data = response.get_json()
        assert "SenaAIgent" in data["service"]
        assert "version" in data


class TestWaterQualityEndpoint:
    """Tests for the water quality analysis endpoint."""

    def test_water_get_returns_200(self, client):
        """Test that GET request returns API documentation."""
        response = client.get("/api/water")
        assert response.status_code == 200

    def test_water_get_returns_documentation(self, client):
        """Test that GET request returns endpoint documentation."""
        response = client.get("/api/water")
        data = response.get_json()
        assert "endpoint" in data
        assert "description" in data
        assert "post_parameters" in data

    def test_water_post_with_valid_data(self, client):
        """Test POST request with valid water quality data."""
        response = client.post(
            "/api/water",
            json={
                "ph": 7.2,
                "turbidity": 2.0,
                "temperature": 22.0,
                "dissolved_oxygen": 8.0,
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "analysis" in data
        assert "quality_score" in data["analysis"]
        assert "quality_label" in data["analysis"]

    def test_water_post_with_empty_body(self, client):
        """Test POST request with empty body uses defaults."""
        response = client.post(
            "/api/water",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

    def test_water_post_with_invalid_ph(self, client):
        """Test POST request with invalid pH value."""
        response = client.post(
            "/api/water",
            json={"ph": "invalid"},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data


class TestImageGenerationEndpoint:
    """Tests for the image generation endpoint."""

    def test_image_get_returns_200(self, client):
        """Test that GET request returns API documentation."""
        response = client.get("/api/image")
        assert response.status_code == 200

    def test_image_get_returns_styles(self, client):
        """Test that GET request returns available styles."""
        response = client.get("/api/image")
        data = response.get_json()
        assert "available_styles" in data
        assert isinstance(data["available_styles"], list)
        assert len(data["available_styles"]) > 0

    def test_image_post_with_valid_prompt(self, client):
        """Test POST request with valid prompt."""
        response = client.post(
            "/api/image",
            json={
                "prompt": "A beautiful sunset over mountains",
                "width": 512,
                "height": 512,
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert ("mock" in data or "engine" in data)

    def test_image_post_without_prompt(self, client):
        """Test POST request without prompt returns error."""
        response = client.post(
            "/api/image",
            json={"width": 512},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "error" in data

    def test_image_post_with_empty_prompt(self, client):
        """Test POST request with empty prompt returns error."""
        response = client.post(
            "/api/image",
            json={"prompt": ""},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_image_post_with_style(self, client):
        """Test POST request with style parameter."""
        response = client.post(
            "/api/image",
            json={
                "prompt": "A serene lake",
                "style": "watercolor",
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data.get("style") == "watercolor"


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_returns_json(self, client):
        """Test that 404 errors return JSON."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data
        assert "available_endpoints" in data
