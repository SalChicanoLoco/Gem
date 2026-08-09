"""
Tests for GemmaAgent class and Gemma API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.gemma_agent import GemmaAgent


@pytest.fixture
def gemma():
    """Create a GemmaAgent instance for testing."""
    return GemmaAgent()


class TestGemmaAgentInit:
    """Tests for GemmaAgent initialization and status."""

    def test_init_defaults(self, gemma):
        """Test default initialization."""
        assert gemma.model_name == "gemma2:2b"
        assert gemma.ollama_host == "http://localhost:11434"
        assert gemma.timeout == 4

    def test_get_status(self, gemma):
        """Test status dictionary output."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": [{"name": "gemma2:2b"}]}
            status = gemma.get_status()
            assert status["agent_id"] == "gemma"
            assert status["model"] == "gemma2:2b"
            assert "status" in status
            assert "hardware_acceleration" in status


class TestGemmaInferenceAndFallback:
    """Tests for Gemma inference generation and fallbacks."""

    def test_generate_fallback(self, gemma):
        """Test text generation fallback when server offline."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Offline")
            res = gemma.generate("Analyze water telemetry quality")
            assert isinstance(res, str)
            assert len(res) > 0

    def test_analyze_telemetry(self, gemma):
        """Test telemetry analysis wrapper."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Telemetry analysis: Water quality normal."}
            data = {"ph": 7.4, "turbidity": 1.2, "temperature": 21.0, "dissolved_oxygen": 8.1}
            result = gemma.analyze_telemetry(data)
            assert result["success"] is True
            assert "analysis" in result

    def test_self_optimize_workflow(self, gemma):
        """Test self-optimization engine."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Workflow optimization: Increase worker pool."}
            queue_status = {"total_tasks": 5, "pending": 2, "processing": 1}
            load_metrics = {"load_score": 45.0, "queue_length": 2}
            res = gemma.self_optimize_workflow(queue_status, load_metrics)
            assert res["success"] is True
            assert "recommendations" in res

    def test_synthesize_aesthetic_prompt(self, gemma):
        """Test aesthetic prompt synthesis."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Impressionist mountain lake scene."}
            prompt = gemma.synthesize_aesthetic_prompt("mountain lake", "impressionist")
            assert isinstance(prompt, str)
            assert len(prompt) > 0
