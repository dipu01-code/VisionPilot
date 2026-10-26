"""
FastAPI Server for VisionPilot.

Exposes REST endpoints for:
- Image upload and analysis
- Health checks
- Agent status
- Trace retrieval
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

from ..core.pipeline import VisionPipeline, ImageRequest, ImageValidationResult
from ..agent.orchestrator import AgentOrchestrator
from .schemas import HealthCheck, AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VisionPilot API",
    description="Agentic Computer Vision system using OpenCV 5",
    version="0.1.0",
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
pipeline = VisionPipeline()
orchestrator = AgentOrchestrator(pipeline)

# Storage for results (in production, use a database)
stored_results: Dict[str, Dict[str, Any]] = {}


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
        opencv_version=pipeline._opencv_version
    )


@app.get("/", response_model=Dict[str, Any], tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "VisionPilot",
        "description": "Agentic Computer Vision system using OpenCV 5",
        "version": "0.1.0",
        "endpoints": {
            "/analyze": "POST - Analyze an image",
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
            # Remove data URL prefix
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


@app.post("/analyze/upload", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_upload(file: UploadFile = File(...)):
    """
    Upload and analyze an image file.

    Accepts multipart/form-data file upload.
    """
    try:
        # Read file contents
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

        # Store result
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

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing uploaded file")
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

    Returns sample images that can be used to demonstrate the system.
    """
    return {
        "demo_images": [
            {
                "id": "empty_scene",
                "description": "Image with no significant objects",
                "url": "/demo/empty_scene.jpg"
            },
            {
                "id": "person_detected",
                "description": "Image with a person in the frame",
                "url": "/demo/person_detected.jpg"
            },
            {
                "id": "multiple_objects",
                "description": "Image with multiple detected objects",
                "url": "/demo/multiple_objects.jpg"
            }
        ],
        "note": "Demo images will be provided as static files in production"
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