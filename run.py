"""
MSME Failure Predictor - Application Entry Point
Run this from the repo root to start the Flask backend server.

Usage:
    python run.py
"""

import sys
import os

# Add repo root to path so backend package resolves correctly
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.app import app
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print("=" * 60)
    print("  MSME FAILURE PREDICTOR — Backend Server")
    print("=" * 60)
    print(f"  Repo root : {_REPO_ROOT}")
    print(f"  Server    : http://0.0.0.0:5000")
    print(f"  Health    : http://localhost:5000/health")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
