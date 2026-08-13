"""
Logging setup for the SenaAIgent processes.

Modules here log through ``logging.getLogger(__name__)``, but nothing configured
the logging system. With no handler and no level, Python falls back to its
"lastResort" handler: WARNING and above reach stderr as a bare message with no
timestamp, level or logger name, and everything at INFO is discarded. The lines
recording which model loaded, which device and precision were chosen, whether a
LoRA applied and whether a step count was clamped were written and never seen.

``configure_logging()`` installs handlers once, at a level from $LOG_LEVEL.
"""

import logging
import os
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Libraries that are useful at WARNING and overwhelming at INFO. Held at WARNING
# unless the caller asks for DEBUG, which is taken to mean "show me everything".
NOISY_LOGGERS = ("diffusers", "transformers", "urllib3", "httpx", "filelock", "PIL")

# Marks handlers this module owns, so reconfiguring replaces ours and leaves
# anyone else's alone — pytest's caplog and a WSGI server's handlers among them.
_OWNED = "_senaaigent_handler"


def _resolve_level(level: Optional[str]):
    """Return (level_int, unknown_name). An unknown name yields INFO, not an error."""
    requested = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    resolved = logging.getLevelName(requested)
    if not isinstance(resolved, int):
        return logging.INFO, requested
    return resolved, None


def _owned_handlers(logger):
    return [h for h in logger.handlers if getattr(h, _OWNED, False)]


def _build_handlers(log_file: Optional[str]):
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers = [logging.StreamHandler(sys.stderr)]

    target = log_file or os.environ.get("LOG_FILE")
    if target:
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        handlers.append(logging.FileHandler(target))

    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _OWNED, True)
    return handlers


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """
    Install a stderr handler, and a file handler when a path is given.

    Args:
        level: Level name such as "DEBUG". Defaults to $LOG_LEVEL, then INFO. An
            unrecognised name falls back to INFO and logs a warning, since a typo
            in an environment variable should not stop a server from booting.
        log_file: Path to also write to; defaults to $LOG_FILE. Parent directories
            are created.
        force: Rebuild handlers even if this already ran. Without it a second call
            is a no-op, so two entry points cannot double every line.

    Returns:
        The root logger.
    """
    root = logging.getLogger()
    already = _owned_handlers(root)
    if already and not force:
        return root

    resolved, unknown = _resolve_level(level)

    for handler in already:
        root.removeHandler(handler)
        handler.close()

    for handler in _build_handlers(log_file):
        root.addHandler(handler)

    root.setLevel(resolved)

    # Set these every time rather than only when pinning, so moving to DEBUG
    # actually lifts a pin left behind by an earlier call.
    library_level = logging.NOTSET if resolved <= logging.DEBUG else max(resolved, logging.WARNING)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(library_level)

    if unknown:
        root.warning("Unknown log level %r; using INFO.", unknown)

    return root


def reset_logging() -> None:
    """Remove this module's handlers and library pins. Intended for tests."""
    root = logging.getLogger()
    for handler in _owned_handlers(root):
        root.removeHandler(handler)
        handler.close()
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)
