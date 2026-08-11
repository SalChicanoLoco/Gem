#!/bin/bash
# Startup Launcher for the SenaAIgent Control Center.
#
# Delegates to scripts/start_stack.py, which starts and then verifies every
# component rather than assuming any of them came up. Starting the API alone left
# chat broken whenever Ollama was not already running, and the failure only
# surfaced later in the UI.
#
#   ./start_gui.sh              start everything and open the dashboard
#   ./start_gui.sh --check      report what is running, start nothing
#   ./start_gui.sh --strict     exit non-zero unless chat works too
#
# Exit code is 0 only when everything required is up.

cd "$(dirname "$0")" || exit 1

PYTHON="./venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

exec "$PYTHON" scripts/start_stack.py "$@"
