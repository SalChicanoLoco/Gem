"""
Tests for GemmaAgent class and Gemma API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.gemma_agent import ConversationStore, GemmaAgent, GemmaUnavailable

# This module tests GemmaAgent itself, so it needs the real methods rather than
# the offline stub conftest installs for everyone else.
pytestmark = pytest.mark.real_gemma


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
        assert gemma.timeout == 120

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

    def test_generate_raises_when_offline(self, gemma):
        """
        An unreachable model must not be answered for.

        This previously returned keyword-matched prose — for a telemetry prompt,
        invented pH and turbidity readings — which a caller could not tell from a
        real answer.
        """
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Offline")
            with pytest.raises(GemmaUnavailable):
                gemma.generate("Analyze water telemetry quality")

    def test_offline_telemetry_reports_failure_rather_than_inventing_readings(self, gemma):
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Offline")
            result = gemma.analyze_telemetry({"ph": 7.4})
        assert result["success"] is False
        assert "error" in result
        assert "analysis" not in result

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

    def test_chat_sends_prior_turns_as_context(self, gemma):
        """
        The point of a conversation is that turn two can refer to turn one, which
        only works if earlier messages are resent. Ollama's API is stateless.
        """
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"message": {"content": "Noted."}}
            gemma.chat("my adapter is sena_style", conversation_id="c1")
            gemma.chat("what is it called?", conversation_id="c1")

            sent = mock_post.call_args.kwargs["json"]["messages"]

        roles = [m["role"] for m in sent]
        assert roles == ["system", "user", "assistant", "user"]
        assert sent[1]["content"] == "my adapter is sena_style"
        assert sent[-1]["content"] == "what is it called?"

    def test_conversations_are_isolated(self, gemma):
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"message": {"content": "ok"}}
            gemma.chat("first", conversation_id="a")
            gemma.chat("second", conversation_id="b")

        assert len(gemma.conversations.get("a")) == 2
        assert gemma.conversations.get("a")[0]["content"] == "first"
        assert gemma.conversations.get("b")[0]["content"] == "second"

    def test_failed_turn_leaves_no_dangling_history(self, gemma):
        """A turn that never got an answer must not pollute later context."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Offline")
            with pytest.raises(GemmaUnavailable):
                gemma.chat("hello", conversation_id="c2")

        assert gemma.conversations.get("c2") == []

    def test_synthesize_aesthetic_prompt(self, gemma):
        """Test aesthetic prompt synthesis."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Impressionist mountain lake scene."}
            prompt = gemma.synthesize_aesthetic_prompt("mountain lake", "impressionist")
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestContextWindow:
    """
    Ollama does not use a model's full context unless asked. Measured with
    gemma2:2b, which declares 8192: a ~4900-token prompt lost its opening at
    num_ctx=2048 and was recalled correctly at 8192.
    """

    def test_context_length_is_read_from_the_model(self, gemma):
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "model_info": {"gemma2.context_length": 8192, "gemma2.embedding_length": 2304}}
            assert gemma.context_length() == 8192

    def test_the_window_is_requested_on_every_call(self, gemma):
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"model_info": {"gemma2.context_length": 8192}}
            options = gemma._options(0.5)
        assert options["num_ctx"] == 8192
        assert options["temperature"] == 0.5

    def test_environment_override_wins(self, gemma, monkeypatch):
        """A larger window costs memory per loaded model, so it stays tunable."""
        monkeypatch.setenv("GEMMA_NUM_CTX", "4096")
        assert gemma.context_length() == 4096

    def test_a_non_numeric_override_is_ignored_rather_than_fatal(self, gemma, monkeypatch):
        monkeypatch.setenv("GEMMA_NUM_CTX", "enormous")
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"model_info": {"gemma2.context_length": 8192}}
            assert gemma.context_length() == 8192

    def test_the_probe_runs_once(self, gemma):
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"model_info": {"gemma2.context_length": 8192}}
            gemma.context_length()
            gemma.context_length()
            gemma.context_length()
            assert mock_post.call_count == 1

    def test_an_unreachable_probe_omits_the_option(self, gemma):
        """Better to let Ollama default than to send a made-up window."""
        with patch("requests.post", side_effect=Exception("offline")):
            assert gemma.context_length() is None
            assert "num_ctx" not in gemma._options(0.7)


class TestConversationStore:
    """History is capped so a long session cannot grow without bound."""

    def test_append_and_get(self):
        store = ConversationStore()
        store.append("c", "user", "hi")
        store.append("c", "assistant", "hello")
        assert [m["role"] for m in store.get("c")] == ["user", "assistant"]

    def test_unknown_conversation_is_empty(self):
        assert ConversationStore().get("nope") == []

    def test_oldest_messages_are_dropped_at_the_cap(self):
        store = ConversationStore(max_messages=4)
        for i in range(6):
            store.append("c", "user", f"m{i}")
        kept = [m["content"] for m in store.get("c")]
        assert kept == ["m2", "m3", "m4", "m5"]

    def test_reset_clears_only_that_conversation(self):
        store = ConversationStore()
        store.append("a", "user", "x")
        store.append("b", "user", "y")
        store.reset("a")
        assert store.get("a") == []
        assert len(store.get("b")) == 1

    def test_returned_history_is_a_copy(self):
        """A caller mutating the returned list must not corrupt stored history."""
        store = ConversationStore()
        store.append("c", "user", "x")
        store.get("c").append({"role": "user", "content": "injected"})
        assert len(store.get("c")) == 1
