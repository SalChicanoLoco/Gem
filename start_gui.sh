#!/bin/bash
# Startup Launcher Script for SenaAIgent Control Center
# Runs CleanHouse pre-flight port sweep, starts API server on Port 5005, and opens GUI dashboard.

echo "============================================================"
echo "🚀 Launching SenaAIgent Pre-AI OS & Control Center"
echo "============================================================"

# Navigate to repository directory
cd "$(dirname "$0")"

# Execute CleanHouse pre-flight port sweep
./venv/bin/python scripts/cleanhouse.py

# Launch web server and open dashboard in browser
echo "🌐 Starting API Server on Port 5005..."
(sleep 2 && open "http://localhost:5005/dashboard") &
PORT=5005 ./venv/bin/python start_api.py
