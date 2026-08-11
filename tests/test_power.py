"""
Tests for the master switch.

The property that matters is that the reported state is measured, not remembered.
A switch that latches to the last command would read ON over a dead model server,
which is exactly the class of defect this project has been removing.
"""

from unittest.mock import patch

import pytest

from agents import power


class _Engine:
    """Stand-in with the one attribute the power module looks at."""

    def __init__(self, loaded=False):
        self.pipe = object() if loaded else None
        self.initialized = loaded
        self.active_lora = None


class TestStatus:
    def test_everything_up_reads_on(self):
        with patch.object(power, "_ollama_models", return_value=["gemma2:2b"]):
            result = power.status([_Engine(loaded=True)])
        assert result["state"] == "on"
        assert result["components"]["model_server"]["up"] is True
        assert result["components"]["model"]["up"] is True

    def test_nothing_up_reads_off(self):
        with patch.object(power, "_ollama_models", return_value=None):
            result = power.status([_Engine(loaded=False)])
        assert result["state"] == "off"

    def test_server_up_without_the_model_is_partial(self):
        """Half a stack is not a working stack, and is not rounded up to on."""
        with patch.object(power, "_ollama_models", return_value=["llama3:8b"]):
            result = power.status([])
        assert result["state"] == "partial"
        assert result["components"]["model"]["up"] is False

    def test_loaded_pipelines_are_reported_but_are_not_required(self):
        """Pipelines load lazily, so their absence must not read as a fault."""
        with patch.object(power, "_ollama_models", return_value=["gemma2:2b"]):
            without = power.status([_Engine(loaded=False)])
            with_pipe = power.status([_Engine(loaded=True)])
        assert without["state"] == "on"
        assert without["components"]["pipelines"]["up"] is False
        assert with_pipe["components"]["pipelines"]["up"] is True

    def test_state_is_measured_not_remembered(self):
        """
        The core guarantee: after a successful power_on, if the server dies the
        very next status call must say so.
        """
        with patch.object(power, "_ollama_models", return_value=["gemma2:2b"]), \
             patch.object(power, "_start_ollama", return_value=True):
            assert power.power_on([])["state"] == "on"

        with patch.object(power, "_ollama_models", return_value=None):
            assert power.status([])["state"] == "off"


class TestPowerOn:
    def test_starts_the_server_when_it_is_down(self):
        calls = []
        models = [None, None, ["gemma2:2b"]]

        def fake_models():
            return models.pop(0) if models else ["gemma2:2b"]

        with patch.object(power, "_ollama_models", side_effect=fake_models), \
             patch.object(power, "_start_ollama", side_effect=lambda: calls.append(1) or True):
            result = power.power_on([], wait_seconds=2)
        assert calls, "should have tried to start the server"
        assert result["state"] == "on"

    def test_a_missing_binary_is_reported_rather_than_hidden(self):
        with patch.object(power, "_ollama_models", return_value=None), \
             patch.object(power, "_start_ollama", return_value=False):
            result = power.power_on([], wait_seconds=1)
        assert result["state"] == "off"
        assert "error" in result
        assert any("not installed" in a for a in result["actions"])

    def test_an_already_running_server_is_not_restarted(self):
        with patch.object(power, "_ollama_models", return_value=["gemma2:2b"]), \
             patch.object(power, "_start_ollama") as start:
            result = power.power_on([])
        start.assert_not_called()
        assert result["state"] == "on"


class TestPowerOff:
    def test_releases_pipelines_and_stops_the_server(self):
        engine = _Engine(loaded=True)
        with patch.object(power, "_ollama_models", side_effect=[["gemma2:2b"], None, None]), \
             patch.object(power, "_stop_ollama", return_value=True):
            result = power.power_off([engine])
        assert engine.pipe is None, "the pipeline should have been released"
        assert result["state"] == "off"

    def test_reports_when_the_server_refuses_to_stop(self):
        """A stop that did not stop must not be reported as success."""
        with patch.object(power, "_ollama_models", return_value=["gemma2:2b"]), \
             patch.object(power, "_stop_ollama", return_value=False):
            result = power.power_off([])
        assert any("did not stop" in a for a in result["actions"])
        assert result["state"] != "off"

    def test_is_safe_to_run_when_already_off(self):
        with patch.object(power, "_ollama_models", return_value=None):
            result = power.power_off([_Engine(loaded=False)])
        assert result["state"] == "off"
        assert any("already stopped" in a for a in result["actions"])
