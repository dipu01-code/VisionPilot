"""
Pydantic schemas for VisionPilot API.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ImageQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    POOR = "poor"
    BLURRY = "blurly"
    DARK = "dark"
    BRIGHT = "bright"
    INVALID = "invalid"


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    opencv_version: str


class DetectionResult(BaseModel):
    """Detection result from image analysis"""
    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]  # x, y, w, h
    center: tuple[float, float]
    area_pixels: int

    class Config:
        json_schema_extra = {
            "example": {
                "label": "person",
                "confidence": 0.85,
                "bounding_box": [100, 50, 120, 300],
                "center": [160.0, 200.0],
                "area_pixels": 36000
            }
        }


class MeasurementResult(BaseModel):
    """Measurement result from geometric analysis"""
    pixel_distance: float
    pixel_area: float
    measurement_unit: str

    class Config:
        json_schema_extra = {
            "example": {
                "pixel_distance": 150.5,
                "pixel_area": 45000.0,
                "measurement_unit": "meters"
            }
        }


class AnalysisResponse(BaseModel):
    """Response from image analysis"""
    request_id: str
    validation: Dict[str, Any]
    detections: List[Dict[str, Any]]
    measurements: List[Dict[str, Any]]
    annoted_image_base64: Optional[str] = None
    decision: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]
    outcome: str
    traces: List[Dict[str, Any]]
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "abc123",
                "validation": {
                    "is_valid": True,
                    "quality": "good",
                    "quality_score": 0.75
                },
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.85,
                        "bounding_box": [100, 50, 120, 300],
                        "center": [160.0, 200.0],
                        "area_pixels": 36000
                    }
                ],
                "measurements": [],
                "annoted_image_base64": "base64...",
                "decision": {
                    "state": "completed",
                    "reason": "Good detections found",
                    "confidence": 0.85,
                    "tool": "store_result",
                    "needs_human_approval": False
                },
                "outcome": "success",
                "traces": [],
                "processing_time_ms": 150.5
            }
        }


class AnalysisRequest(BaseModel):
    """Request for image analysis"""
    image_data: str = Field(..., description="Base64 encoded image data")
    filename: str = Field(..., description="Original filename")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA...",
                "filename": "test_image.jpg",
                "request_id": "optional-request-id"
            }
        }