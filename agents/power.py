"""
Master power for the AI stack.

The switch controls the expensive subsystems — the local model server and the
loaded diffusion pipelines — not the web server itself. That boundary is
deliberate: the dashboard is served by the web server, so a switch that killed it
would kill the page holding the switch, and there would be nothing left to switch
back on. The panel stays powered; the stack behind it is what turns off.

Every reading here is measured at the moment it is asked for. Nothing caches "we
turned it on" and reports that later as if it were still true, because a switch
that shows ON while the model is unreachable is the exact class of defect this
project has spent its time removing.
"""

import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

from . import memory_manager

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
REQUIRED_MODEL = os.environ.get("GEMMA_MODEL", "gemma2:2b")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _ollama_models() -> Optional[List[str]]:
    """Models Ollama has, or None when it is not answering."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
        if resp.status_code != 200:
            return None
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return None


def _start_ollama() -> bool:
    if shutil.which("ollama") is None:
        logger.error("Cannot start the model server: ollama is not on PATH.")
        return False
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "ollama.log"), "ab") as log:
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, start_new_session=True)
    return True


def _stop_ollama() -> bool:
    """Stop the model server. Returns whether it is actually gone afterwards."""
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True, check=False)
    for _ in range(20):
        if _ollama_models() is None:
            return True
        time.sleep(0.25)
    return _ollama_models() is None


def status(engines: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Measured state of every subsystem the switch governs.

    ``state`` is on only when everything required is genuinely up; partial is
    reported as partial rather than rounded up to on.
    """
    models = _ollama_models()
    model_ready = bool(models) and any(REQUIRED_MODEL in name for name in models)
    loaded = [type(e).__name__ for e in (engines or []) if getattr(e, "pipe", None) is not None]

    components = {
        "model_server": {
            "up": models is not None,
            "detail": f"{OLLAMA_HOST}" if models is not None else "not reachable",
        },
        "model": {
            "up": model_ready,
            "detail": REQUIRED_MODEL if model_ready else f"{REQUIRED_MODEL} not available",
        },
        "pipelines": {
            "up": bool(loaded),
            "detail": ", ".join(loaded) if loaded else "none loaded (they load on first use)",
        },
    }

    required_up = components["model_server"]["up"] and components["model"]["up"]
    if required_up:
        state = "on"
    elif any(c["up"] for c in components.values()):
        state = "partial"
    else:
        state = "off"

    return {
        "success": True,
        "state": state,
        "components": components,
        "memory": memory_manager.snapshot(),
        "measured_at": time.time(),
    }


def power_on(engines: Optional[List[Any]] = None, wait_seconds: int = 25) -> Dict[str, Any]:
    """
    Bring the stack up and report what actually came up.

    Pipelines are left to load lazily on first use rather than being warmed here:
    loading SD 1.5 takes seconds and the video pipelines take considerably longer,
    so warming them would make the switch feel broken while nothing appeared to
    happen.
    """
    actions: List[str] = []

    if _ollama_models() is None:
        if _start_ollama():
            actions.append("started the model server")
            deadline = time.time() + wait_seconds
            while time.time() < deadline and _ollama_models() is None:
                time.sleep(0.5)
        else:
            actions.append("could not start the model server: ollama is not installed")
    else:
        actions.append("model server was already running")

    result = status(engines)
    result["actions"] = actions
    result["requested"] = "on"
    if result["state"] != "on":
        result["error"] = "the stack did not fully come up; see components"
    logger.info("Power on requested: %s (state=%s)", "; ".join(actions), result["state"])
    return result


def power_off(engines: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Release loaded pipelines and stop the model server.

    The web server keeps running, so the panel survives to switch back on.
    """
    actions: List[str] = []

    released = memory_manager.release(*(engines or []))
    if released["released"]:
        actions.append(f"released {', '.join(released['released'])}"
                       + (f", freeing {released['rss_mb_freed']}MB" if released["rss_mb_freed"] else ""))
    else:
        actions.append("no pipelines were loaded")

    if _ollama_models() is not None:
        actions.append("stopped the model server" if _stop_ollama()
                       else "the model server did not stop")
    else:
        actions.append("model server was already stopped")

    result = status(engines)
    result["actions"] = actions
    result["requested"] = "off"
    result["released"] = released
    logger.info("Power off requested: %s (state=%s)", "; ".join(actions), result["state"])
    return result
