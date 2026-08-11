"""
Tests for the stack launcher.

The launcher's value is that it tells the truth about what came up, so these
assert the reporting and the exit code rather than that a process spawned.
"""

import pytest
from unittest.mock import patch

from scripts import start_stack
from scripts.start_stack import FAIL, OK, WARN, Report


class TestReport:
    def test_worst_is_ok_when_all_ok(self):
        r = Report()
        r.add("a", OK)
        r.add("b", OK)
        assert r.worst() == OK

    def test_a_single_warning_degrades_the_whole_report(self):
        r = Report()
        r.add("a", OK)
        r.add("b", WARN)
        assert r.worst() == WARN

    def test_failure_outranks_warning(self):
        r = Report()
        r.add("a", WARN)
        r.add("b", FAIL)
        r.add("c", OK)
        assert r.worst() == FAIL

    def test_empty_report_is_ok(self):
        assert Report().worst() == OK


class TestWaitUntil:
    def test_returns_first_truthy_value(self):
        values = iter([None, None, "ready"])
        assert start_stack.wait_until(lambda: next(values), timeout=5, interval=0) == "ready"

    def test_returns_none_once_the_deadline_passes(self):
        assert start_stack.wait_until(lambda: None, timeout=0.05, interval=0.01) is None


class TestEnsureOllama:
    def test_reports_ok_when_already_serving_the_model(self):
        report = Report()
        with patch.object(start_stack, "ollama_models", return_value=["gemma2:2b"]):
            assert start_stack.ensure_ollama(report, start=False) is True
        assert report.worst() == OK

    def test_missing_model_is_a_failure_with_the_pull_command(self):
        """Ollama running without the model is a different fix from Ollama being down."""
        report = Report()
        with patch.object(start_stack, "ollama_models", return_value=["llama3:8b"]):
            assert start_stack.ensure_ollama(report, start=False) is False
        detail = [d for name, _, d in report.rows if name == "model"][0]
        assert "ollama pull" in detail
        assert "llama3:8b" in detail  # says what is actually installed

    def test_unreachable_server_is_reported_not_guessed(self):
        report = Report()
        with patch.object(start_stack, "ollama_models", return_value=None):
            assert start_stack.ensure_ollama(report, start=False) is False
        assert report.worst() == FAIL

    def test_missing_binary_is_distinguished_from_a_down_server(self):
        report = Report()
        with patch.object(start_stack, "ollama_models", return_value=None), \
             patch.object(start_stack, "start_ollama", side_effect=FileNotFoundError("nope")):
            assert start_stack.ensure_ollama(report, start=True) is False
        detail = [d for name, _, d in report.rows if name == "ollama"][0]
        assert "not installed" in detail


class TestCheckChat:
    def test_available_model_reports_ok(self):
        report = Report()
        start_stack.check_chat(report, {"gemma_status": {"available": True, "model": "gemma2:2b"}})
        assert report.worst() == OK

    def test_api_that_cannot_see_the_model_is_a_warning(self):
        """Both processes running does not mean the API can reach the model."""
        report = Report()
        start_stack.check_chat(report, {"gemma_status": {"available": False, "status": "fallback_mode"}})
        assert report.worst() == WARN

    def test_missing_status_block_does_not_crash(self):
        report = Report()
        start_stack.check_chat(report, {})
        assert report.worst() == WARN


class TestExitCode:
    """Exit code is the part a script or a person actually branches on."""

    def _run(self, argv, health, chat_ready):
        with patch.object(start_stack, "check_diffusion", return_value=OK), \
             patch.object(start_stack, "ensure_ollama", return_value=chat_ready), \
             patch.object(start_stack, "free_ports"), \
             patch.object(start_stack, "ensure_api", return_value=health), \
             patch("subprocess.run"):
            return start_stack.main(argv)

    def test_zero_when_everything_is_up(self):
        health = {"gemma_status": {"available": True, "model": "gemma2:2b"}}
        assert self._run(["--no-browser"], health, True) == 0

    def test_zero_when_only_chat_is_down_by_default(self):
        """Diffusion works without a model server, so this is degraded, not broken."""
        health = {"gemma_status": {"available": False, "status": "fallback_mode"}}
        assert self._run(["--no-browser"], health, False) == 0

    def test_nonzero_when_chat_is_down_under_strict(self):
        health = {"gemma_status": {"available": False, "status": "fallback_mode"}}
        assert self._run(["--no-browser", "--strict"], health, False) == 1

    def test_nonzero_when_the_api_never_came_up(self):
        assert self._run(["--no-browser"], None, True) == 1
