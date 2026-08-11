#!/usr/bin/env python3
"""
Bring up the whole SenaAIgent stack and report honestly on what came up.

Starting the API alone is not enough. Chat needs a local Ollama serving a pulled
model, and when that is missing the failure surfaces much later as an error in the
UI rather than at launch. This checks every dependency up front, starts what it
can, waits for each to actually answer, and prints a status table.

Exit code is 0 when everything required is up, 1 otherwise. Ollama is optional by
default because diffusion works without it; --strict makes it required.

  python scripts/start_stack.py                # start everything, warn on chat
  python scripts/start_stack.py --strict       # fail unless chat works too
  python scripts/start_stack.py --check        # report status, start nothing
  python scripts/start_stack.py --no-browser   # skip opening the dashboard
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_API_PORT = 5005
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
REQUIRED_MODEL = os.environ.get("GEMMA_MODEL", "gemma2:2b")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_ROOT, "logs")

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}


class Report:
    """Collected component statuses, so the summary reflects every check."""

    def __init__(self):
        self.rows = []

    def add(self, component, state, detail=""):
        self.rows.append((component, state, detail))
        print(f"[{_MARK[state]}] {component:<22} {detail}")
        return state

    def worst(self):
        states = {state for _, state, _ in self.rows}
        if FAIL in states:
            return FAIL
        return WARN if WARN in states else OK


def http_json(url, timeout=2.0):
    """GET a URL and parse JSON, or return None if it is not reachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def wait_until(predicate, timeout, interval=0.5):
    """Poll predicate until it returns truthy. Returns the value, or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def ollama_models(host=OLLAMA_HOST):
    """Names of models Ollama has locally, or None when it is not running."""
    data = http_json(f"{host}/api/tags", timeout=2.0)
    if data is None:
        return None
    return [m.get("name", "") for m in data.get("models", [])]


def start_ollama(log_dir=LOG_DIR):
    """
    Launch `ollama serve` detached, returning the log path.

    Raises FileNotFoundError when the binary is not installed, which is a
    different problem from the server being down and is reported differently.
    """
    if shutil.which("ollama") is None:
        raise FileNotFoundError("ollama binary not found on PATH")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ollama.log")
    with open(log_path, "ab") as log:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=log, stderr=log,
            start_new_session=True,
        )
    return log_path


def ensure_ollama(report, start=True):
    """Make the local model server reachable, and confirm the model is pulled."""
    models = ollama_models()

    if models is None and start:
        try:
            log_path = start_ollama()
        except FileNotFoundError:
            report.add("ollama", FAIL,
                       "not installed - `brew install ollama`; chat will be unavailable")
            return False
        report.add("ollama", OK, f"starting, logging to {os.path.relpath(log_path, REPO_ROOT)}")
        models = wait_until(ollama_models, timeout=20)

    if models is None:
        report.add("ollama", FAIL, f"not reachable at {OLLAMA_HOST}; chat will be unavailable")
        return False

    report.add("ollama", OK, f"serving at {OLLAMA_HOST}")

    if not any(REQUIRED_MODEL in name for name in models):
        have = ", ".join(models) or "none"
        report.add("model", FAIL,
                   f"{REQUIRED_MODEL} not pulled (have: {have}) - run `ollama pull {REQUIRED_MODEL}`")
        return False

    report.add("model", OK, REQUIRED_MODEL)
    return True


def free_ports(report, ports):
    """Clear stale listeners so the API is not silently pushed to another port."""
    try:
        from scripts.cleanhouse import cleanhouse
    except ImportError as e:
        report.add("ports", WARN, f"sweep unavailable: {e}")
        return
    result = cleanhouse(ports=ports)
    cleaned = result.get("cleaned_processes") or []
    detail = f"cleared {cleaned}" if cleaned else "no stale listeners"
    report.add("ports", OK, detail)


def api_health(port):
    return http_json(f"http://localhost:{port}/", timeout=2.0)


def start_api(port, log_dir=LOG_DIR):
    """Launch the Flask API detached, returning (process, log path)."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "api.log")
    env = dict(os.environ, PORT=str(port), PYTHONPATH=REPO_ROOT)
    python = os.path.join(REPO_ROOT, "venv", "bin", "python")
    if not os.path.exists(python):
        python = sys.executable
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [python, os.path.join(REPO_ROOT, "start_api.py")],
            stdout=log, stderr=log, cwd=REPO_ROOT, env=env,
            start_new_session=True,
        )
    return proc, log_path


def ensure_api(report, port, start=True, timeout=90):
    """Make the API answer on its port, reporting what it says about itself."""
    health = api_health(port)

    if health is None and start:
        proc, log_path = start_api(port)
        rel = os.path.relpath(log_path, REPO_ROOT)
        report.add("api", OK, f"starting on :{port}, logging to {rel}")
        health = wait_until(lambda: api_health(port), timeout=timeout, interval=1.0)
        if health is None and proc.poll() is not None:
            report.add("api", FAIL, f"process exited with code {proc.returncode}; see {rel}")
            return None

    if health is None:
        # Distinguish "we waited and it never came up" from "we only looked once",
        # which is all --check does.
        detail = (f"no response on :{port} within {timeout}s" if start
                  else f"not running on :{port}")
        report.add("api", FAIL, detail)
        return None

    report.add("api", OK, f"http://localhost:{port} ({health.get('service', 'unknown service')})")
    return health


def check_diffusion(report):
    """Report the image engine's real device, not an assumed one."""
    try:
        from agents.diffusion_engine import DIFFUSERS_AVAILABLE, MPS_AVAILABLE, TORCH_AVAILABLE
    except ImportError as e:
        return report.add("diffusion", FAIL, f"import failed: {e}")

    if not (TORCH_AVAILABLE and DIFFUSERS_AVAILABLE):
        return report.add("diffusion", FAIL,
                          "torch/diffusers missing - `pip install -r requirements.txt`")
    if not MPS_AVAILABLE:
        return report.add("diffusion", WARN, "MPS unavailable; renders will fall back to CPU")
    return report.add("diffusion", OK, "torch + diffusers, Metal GPU (MPS)")


def check_chat(report, health):
    """Confirm the API itself can see the model, not just that both are running."""
    status = (health or {}).get("gemma_status") or {}
    if status.get("available"):
        return report.add("chat", OK, f"{status.get('model', 'model')} reachable from the API")
    return report.add("chat", WARN,
                      f"API reports {status.get('status', 'unavailable')}; chat will show an error")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start and verify the SenaAIgent stack.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", DEFAULT_API_PORT)))
    parser.add_argument("--strict", action="store_true",
                        help="treat a missing model server as a failure")
    parser.add_argument("--check", action="store_true",
                        help="report status without starting anything")
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    args = parser.parse_args(argv)

    starting = not args.check
    print("=" * 64)
    print("  SenaAIgent stack " + ("check" if args.check else "startup"))
    print("=" * 64)

    report = Report()
    check_diffusion(report)
    chat_ready = ensure_ollama(report, start=starting)
    if starting:
        free_ports(report, [args.port])
    health = ensure_api(report, args.port, start=starting)
    if health is not None:
        check_chat(report, health)

    print("-" * 64)
    worst = report.worst()
    api_up = health is not None
    ok = api_up and (chat_ready or not args.strict) and worst != FAIL

    if api_up and worst == OK:
        print("All components up.")
    elif api_up:
        print("API is up. Some components are degraded - see the rows marked above.")
    else:
        print("The API did not come up. The stack is not usable.")

    if api_up:
        print(f"  Dashboard : http://localhost:{args.port}/dashboard")
        print(f"  Chat      : http://localhost:{args.port}/static/chat.html")
        if not args.no_browser and not args.check:
            subprocess.run(["open", f"http://localhost:{args.port}/dashboard"], check=False)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
