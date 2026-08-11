"""
Tests for logging setup.

The bug being guarded against is silence: the agents logged throughout, but with
no configuration Python discarded everything below WARNING, so the lines saying
which model and precision were chosen never appeared.
"""

import logging

import pytest

from agents.logging_config import (
    NOISY_LOGGERS,
    configure_logging,
    reset_logging,
)


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Logging is global state; leave it exactly as found."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_noisy = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    yield
    reset_logging()
    root.handlers = saved_handlers
    root.setLevel(saved_level)
    for name, level in saved_noisy.items():
        logging.getLogger(name).setLevel(level)


class TestConfigureLogging:
    def test_info_records_are_emitted_after_configuring(self):
        """The whole point: info from the engines must survive."""
        configure_logging(level="INFO", force=True)
        logger = logging.getLogger("agents.diffusion_engine")
        assert logger.isEnabledFor(logging.INFO)

    def test_a_handler_is_installed(self):
        configure_logging(level="INFO", force=True)
        assert logging.getLogger().handlers, "root logger left with no handler"

    def test_foreign_handlers_are_left_alone(self):
        """
        Reconfiguring must not evict handlers owned by someone else. Stripping
        every root handler removed pytest's caplog handler, and would do the same
        to a WSGI server's.
        """
        root = logging.getLogger()
        foreign = logging.NullHandler()
        root.addHandler(foreign)

        configure_logging(level="INFO", force=True)
        configure_logging(level="DEBUG", force=True)

        assert foreign in root.handlers

    def test_level_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        configure_logging(force=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_explicit_level_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        configure_logging(level="ERROR", force=True)
        assert logging.getLogger().level == logging.ERROR

    def test_unknown_level_falls_back_instead_of_raising(self, monkeypatch):
        """A typo in an env var must not stop a server from booting."""
        monkeypatch.setenv("LOG_LEVEL", "CHATTY")
        configure_logging(force=True)
        assert logging.getLogger().level == logging.INFO

    def test_repeat_calls_do_not_stack_handlers(self):
        """Two entry points importing each other must not double every line."""
        configure_logging(level="INFO", force=True)
        count = len(logging.getLogger().handlers)
        configure_logging(level="INFO")
        configure_logging(level="INFO")
        assert len(logging.getLogger().handlers) == count

    def test_force_replaces_rather_than_appends(self):
        configure_logging(level="INFO", force=True)
        count = len(logging.getLogger().handlers)
        configure_logging(level="DEBUG", force=True)
        assert len(logging.getLogger().handlers) == count


class TestLibraryNoise:
    def test_noisy_libraries_are_pinned_above_info(self):
        configure_logging(level="INFO", force=True)
        assert logging.getLogger("diffusers").getEffectiveLevel() >= logging.WARNING

    def test_debug_lifts_a_pin_left_by_an_earlier_call(self):
        """
        The pin must be re-evaluated on every call. Setting it only when pinning
        left diffusers stuck at WARNING even after asking for DEBUG.
        """
        configure_logging(level="INFO", force=True)
        configure_logging(level="DEBUG", force=True)
        assert logging.getLogger("diffusers").getEffectiveLevel() == logging.DEBUG


class TestFileOutput:
    def test_writes_to_a_file_when_asked(self, tmp_path):
        target = tmp_path / "nested" / "run.log"
        configure_logging(level="INFO", log_file=str(target), force=True)
        logging.getLogger("agents.test").info("written to disk")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert target.exists()
        assert "written to disk" in target.read_text()

    def test_log_file_env_var_is_honoured(self, tmp_path, monkeypatch):
        target = tmp_path / "from_env.log"
        monkeypatch.setenv("LOG_FILE", str(target))
        configure_logging(level="INFO", force=True)
        logging.getLogger("agents.test").info("env routed")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "env routed" in target.read_text()
