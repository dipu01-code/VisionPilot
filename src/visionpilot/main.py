#!/usr/bin/env python3
"""
Main entry point for VisionPilot API server.

Usage:
    python -m visionpilot.main
    uvicorn visionpilot.main:app --host 0.0.0.0 --port 8000
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionpilot.api.video_server import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "visionpilot.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )