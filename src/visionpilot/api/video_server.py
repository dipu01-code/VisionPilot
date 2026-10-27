"""
Enhanced FastAPI Server for VisionPilot with Video Support.

Extends the base VisionPilot API to handle video analysis:
- Video upload and processing
- Frame extraction and annotation
- Temporal analysis results
- Video-specific endpoints and analytics
"""

import base64
import io
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.pipeline import VisionPipeline, ImageRequest
from ..core.video_pipeline import VideoPipeline as EnhancedVideoPipeline
from ..agent.orchestrator import AgentOrchestrator
from .schemas import HealthCheck, AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VisionPilot API with Video Support",
    description="Agentic Computer Vision system using OpenCV 5 with video processing",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
image_pipeline = VisionPipeline()
video_pipeline = EnhancedVideoPipeline()
orchestrator = AgentOrchestrator(image_pipeline)

# Storage for results
stored_results: Dict[str, Dict[str, Any]] = {}


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="0.2.0",
        timestamp=datetime.utcnow().isoformat(),
        opencv_version=image_pipeline._opencv_version
    )


@app.get("/", response_model=Dict[str, Any], tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "VisionPilot",
        "description": "Agentic Computer Vision system with video support using OpenCV 5",
        "version": "0.2.0",
        "endpoints": {
            "/analyze": "POST - Analyze an image",
            "/analyze/upload": "POST - Upload and analyze an image file",
            "/video/analyze": "POST - Analyze a video",
            "/video/analyze/upload": "POST - Upload and analyze a video file",
            "/health": "GET - Health check",
            "/results/{request_id}": "GET - Get analysis results",
            "/traces": "GET - Get recent traces",
            "/demo": "GET - List demo inputs"
        }
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_image(request: AnalysisRequest = Body(...)):
    """
    Analyze an image through the agentic vision pipeline.

    Accepts base64-encoded image data and returns:
    - Detection results with confidence scores
    - Geometric measurements
    - Annotated image
    - Agent decisions and traces
    """
    try:
        # Decode image data
        if request.image_data.startswith("data:image"):
            image_data = base64.b64decode(request.image_data.split(",", 1)[1])
        else:
            image_data = base64.b64decode(request.image_data)

        request_id = request.request_id or str(uuid.uuid4())

        # Process through agentic loop
        result = orchestrator.process_image(
            image_data=image_data,
            filename=request.filename,
            request_id=request_id
        )

        # Store result for later retrieval
        stored_results[request_id] = result

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(result)

    except Exception as e:
        logger.exception("Error processing image")
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")


@app.post("/analyze/upload", tags=["Analysis"])
async def analyze_upload(file: UploadFile = File(...)):
    """
    Upload and analyze an image file.

    Accepts multipart/form-data file upload.
    """
    try:
        content = await file.read()
        image_data = content

        # Validate file type
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {content_type}. Expected image."
            )

        request_id = str(uuid.uuid4())

        # Process through agentic loop
        result = orchestrator.process_image(
            image_data=image_data,
            filename=file.filename or "uploaded_image",
            request_id=request_id
        )

        stored_results[request_id] = result

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing uploaded file")
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")


@app.post("/video/analyze", tags=["Video Analysis"])
async def analyze_video(request: AnalysisRequest = Body(...)):
    """
    Analyze a video through the enhanced agentic vision pipeline.

    Accepts base64-encoded video data and returns:
    - Frame-by-frame detection results with timestamps
    - Temporal measurements
    - Annotated video
    - Agent decisions and traces
    """
    try:
        # Decode video data
        if request.image_data.startswith("data:video"):
            video_data = base64.b64decode(request.image_data.split(",", 1)[1])
        else:
            video_data = base64.b64decode(request.image_data)

        request_id = request.request_id or str(uuid.uuid4())

        # Process through video agentic loop
        result = orchestrator.process_video(
            image_data=video_data,
            filename=request.filename,
            request_id=request_id
        )

        stored_results[request_id] = result

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(result)

    except Exception as e:
        logger.exception("Error processing video")
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")


@app.post("/video/analyze/upload", tags=["Video Analysis"])
async def analyze_video_upload(file: UploadFile = File(...)):
    """
    Upload and analyze a video file.

    Accepts multipart/form-data file upload for video analysis.
    """
    try:
        content = await file.read()
        video_data = content

        # Validate file type
        content_type = file.content_type
        if not content_type or not content_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {content_type}. Expected video."
            )

        request_id = str(uuid.uuid4())

        # Process through video agentic loop
        result = orchestrator.process_video(
            image_data=video_data,
            filename=file.filename or "uploaded_video",
            request_id=request_id
        )

        stored_results[request_id] = result

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing uploaded video")
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")


@app.get("/results/{request_id}", response_model=Dict[str, Any], tags=["Results"])
async def get_result(request_id: str):
    """Get analysis results by request ID"""
    if request_id not in stored_results:
        raise HTTPException(status_code=404, detail="Result not found")

    return stored_results[request_id]


@app.get("/traces", response_model=List[Dict[str, Any]], tags=["Trace"])
async def get_traces(count: int = Query(10, ge=1, le=100)):
    """Get recent agent traces"""
    return orchestrator.get_last_traces(count)


@app.get("/demo", response_model=Dict[str, Any], tags=["Demo"])
async def list_demo_inputs():
    """
    List available demo inputs for testing.

    Returns sample images and videos that can be used to demonstrate the system.
    """
    return {
        "demo_items": [
            {
                "id": "empty_scene",
                "type": "image",
                "description": "Image with no significant objects",
                "url": "/demo/empty_scene.jpg"
            },
            {
                "id": "person_detected",
                "type": "image",
                "description": "Image with a person in the frame",
                "url": "/demo/person_detected.jpg"
            },
            {
                "id": "multiple_objects",
                "type": "image",
                "description": "Image with multiple detected objects",
                "url": "/demo/multiple_objects.jpg"
            },
            {
                "id": "sample_video",
                "type": "video",
                "description": "Sample video with objects moving",
                "url": "/demo/sample_video.mp4"
            }
        ],
        "note": "Demo images and videos will be provided as static files in production"
    }


@app.post("/approve/{approval_id}", response_model=Dict[str, Any], tags=["Approval"])
async def approve_result(approval_id: str, decision: str = Body(..., embed=True)):
    """
    Respond to a human approval request.

    This allows the agentic loop to continue after human input.
    """
    result = orchestrator.respond_to_approval(approval_id, decision, "operator")

    if result["status"] == "approval_not_found":
        raise HTTPException(status_code=404, detail="Approval request not found")

    return result


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )