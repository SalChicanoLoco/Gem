#!/usr/bin/env python3
"""
Script to start the SenaAIgent API server for local testing.
This script ensures the Python path is set correctly.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set PYTHONPATH environment variable for subprocess compatibility
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')

# Without this the agents' logger.info calls are discarded and their warnings
# arrive unformatted, so the server gives no account of what it is doing.
from agents.logging_config import configure_logging
configure_logging()

# Run CleanHouse Pre-Flight Sweep
try:
    from scripts.cleanhouse import cleanhouse
    print("🧹 CleanHouse: Running pre-flight port sweep...")
    ch_report = cleanhouse(ports=[5000, 5005, 8000, 8080])
    if ch_report["cleaned_processes"]:
        print(f"🧹 Cleaned stale webserver processes: {ch_report['cleaned_processes']}")
    default_port = ch_report["recommended_port"]
except Exception as e:
    default_port = 5005

# Get port from environment or use CleanHouse recommended clean port
port = int(os.environ.get('PORT', default_port))
debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

print("=" * 60)
print("  Starting SenaAIgent API Server")
print("=" * 60)
print(f"Project root: {project_root}")
print(f"Port: {port}")
print(f"Debug mode: {debug}")
print()
print("Available endpoints:")
print(f"  - Health check:        http://localhost:{port}/")
print(f"  - OpenAI IDE Gateway: http://localhost:{port}/v1/chat/completions")
print(f"  - OpenAI Models List:  http://localhost:{port}/v1/models")
print(f"  - Coding Agent API:    http://localhost:{port}/api/coder")
print(f"  - PyTorch MPS Diffusion:http://localhost:{port}/api/image/diffusion")
print(f"  - Water quality:       http://localhost:{port}/api/water")
print(f"  - Image generation:    http://localhost:{port}/api/image")
print(f"  - Art analysis:        http://localhost:{port}/api/art")
print(f"  - Orchestrator:        http://localhost:{port}/api/orchestrator")
print(f"  - Dashboard:           http://localhost:{port}/dashboard")
print()
print("Press Ctrl+C to stop the server")
print("=" * 60)
print()

# Import and run the Flask app
try:
    from api.app import app
    app.run(host='0.0.0.0', port=port, debug=debug)
except ImportError as e:
    print(f"Error: Failed to import API application: {e}")
    print()
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)
