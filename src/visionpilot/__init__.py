"""VisionPilot - An Agentic Computer Vision System for OpenCV AI Competition 2026"""

__version__ = "0.1.0"
__author__ = "VisionPilot Team"

from .core.pipeline import VisionPipeline, ImageRequest, ImageResponse
from .agent.orchestrator import AgentOrchestrator
from .api.server import app

__all__ = ["VisionPipeline", "ImageRequest", "ImageResponse", "AgentOrchestrator", "app"]