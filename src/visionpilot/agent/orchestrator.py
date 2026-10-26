"""
Agent Orchestrator - The core agentic vision loop.

Implements the perception-decision-action cycle:
Input -> OpenCV 5 Perception -> Decision Engine -> Actions
Actions can trigger reprocessing, approvals, or storage.

The orchestrator maintains state and produces structured traces.
"""

import uuid
import time
import base64
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.pipeline import VisionPipeline, ImageRequest, ImageResponse
from .types import AgentState, AgentDecision, AgentTrace, ToolType
from .decision_engine import DecisionEngine

logger = logging.getLogger(__name__)


@dataclass
class HumanApproval:
    """Record of a human approval request and response"""
    requested_at: float
    requested_by: str
    message: str
    decision: Optional[str] = None  # "approve" or "reject" or None
    response_at: Optional[float] = None
    approver: Optional[str] = None


class AgentOrchestrator:
    """
    Main orchestrator for the agentic vision system.

    Coordinates the perception-decision-action loop:
    1. Receives image input
    2. Runs OpenCV 5 perception pipeline
    3. Interprets visual evidence
    4. Makes decisions using deterministic rules
    5. Executes actions (process, request approval, store, etc.)
    6. Produces structured traces
    """

    def __init__(self, pipeline: Optional[VisionPipeline] = None):
        self.pipeline = pipeline or VisionPipeline()
        self.decision_engine = DecisionEngine()
        self.traces: List[AgentTrace] = []
        self.current_state = AgentState.IDLE
        self.retry_count = 0
        self.max_retries = 3

        # Human approval tracking
        self.pending_approvals: Dict[str, HumanApproval] = {}

    def process_image(self,
                     image_data: bytes,
                     filename: str,
                     request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image through the agentic vision loop.

        This is the main entry point that produces:
        - A decision based on OpenCV 5 analysis
        - Structured trace records
        - The final processed result

        Returns a dictionary with the complete result including traces.
        """
        request_id = request_id or str(uuid.uuid4())
        observation_id = str(uuid.uuid4())
        all_traces: List[Dict[str, Any]] = []

        # Initialize state
        self.current_state = AgentState.RECEIVED
        self.retry_count = 0
        trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            request_id=request_id,
            observation_id=observation_id,
            timestamp=time.time(),
            opencv_operation="request_received",
            visual_findings={"filename": filename}
        )
        all_traces.append(trace.to_dict())

        # Create processing request
        request = ImageRequest(
            request_id=request_id,
            image_data=image_data,
            filename=filename
        )

        # Run the vision pipeline
        self.current_state = AgentState.ANALYZING
        response = self._run_vision_pipeline(request, observation_id, all_traces)

        # Make a decision based on results
        self.current_state = AgentState.DECIDING
        decision = self.decision_engine.decide(response)
        decision.decision_id = str(uuid.uuid4())

        decision_trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            request_id=request_id,
            observation_id=observation_id,
            timestamp=time.time(),
            visual_findings={
                "detection_count": len(response.detections),
                "detection_labels": [d.label for d in response.detections],
                "quality_score": response.validation.quality_score
            },
            confidence_indicators={
                "quality": response.validation.quality_score,
                "avg_confidence": sum(d.confidence for d in response.detections) / len(response.detections) if response.detections else 0
            },
            decision=decision,
            tool_arguments=decision.tool_args if decision.tool else None
        )
        all_traces.append(decision_trace.to_dict())

        # Execute the action
        result = self._execute_action(response, decision, request_id, observation_id, all_traces)

        self.current_state = AgentState.COMPLETED if decision.state == AgentState.COMPLETED else AgentState.FAILED
        self._store_trace(result.get("traces", []))

        return result

    def _run_vision_pipeline(self,
                            request: ImageRequest,
                            observation_id: str,
                            traces: List[Dict[str, Any]]) -> ImageResponse:
        """Run the OpenCV 5 vision pipeline and record traces"""
        response = self.pipeline.process_request(request)

        # Add trace for OpenCV operations
        for event in response.trace_events:
            trace = AgentTrace(
                trace_id=str(uuid.uuid4()),
                request_id=request.request_id,
                observation_id=observation_id,
                timestamp=time.time(),
                opencv_operation=event.get("operation"),
                visual_findings={"details": event},
                confidence_indicators={"quality_score": response.validation.quality_score} if response.validation else None
            )
            traces.append(trace.to_dict())

        return response

    def _execute_action(self,
                       response: ImageResponse,
                       decision: AgentDecision,
                       request_id: str,
                       observation_id: str,
                       traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the agent's decision.

        Returns the action result and updates traces.
        """
        tool_result = None

        if decision.tool:
            tool_result = self._invoke_tool(decision.tool, decision.tool_args, response)

            # Record tool execution in traces
            tool_trace = AgentTrace(
                trace_id=str(uuid.uuid4()),
                request_id=request_id,
                observation_id=observation_id,
                timestamp=time.time(),
                tool_arguments=decision.tool_args,
                tool_result=tool_result
            )
            traces.append(tool_trace.to_dict())

        # Determine final outcome
        if decision.needs_human_approval:
            outcome = "pending_approval"
        elif decision.state == AgentState.COMPLETED:
            outcome = "success"
        elif decision.state == AgentState.REJECTED:
            outcome = "rejected"
        else:
            outcome = "failed"

        # Record final trace
        final_trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            request_id=request_id,
            observation_id=observation_id,
            timestamp=time.time(),
            final_outcome=outcome
        )
        traces.append(final_trace.to_dict())

        return {
            "request_id": request_id,
            "response": {
                "validation": {
                    "is_valid": response.validation.is_valid,
                    "quality": response.validation.quality.value,
                    "quality_score": response.validation.quality_score
                },
                "detections": [
                    {
                        "label": d.label,
                        "confidence": d.confidence,
                        "bounding_box": d.bounding_box,
                        "center": d.center,
                        "area_pixels": d.area_pixels
                    }
                    for d in response.detections
                ],
                "measurements": [
                    {
                        "pixel_distance": m.pixel_distance,
                        "pixel_area": m.pixel_area,
                        "measurement_unit": m.measurement_unit
                    }
                    for m in response.measurements
                ],
                "annoted_image_base64": response.annotated_image_base64
            },
            "decision": {
                "state": decision.state.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "tool": decision.tool.value if decision.tool else None,
                "needs_human_approval": decision.needs_human_approval
            },
            "tool_result": tool_result,
            "outcome": outcome,
            "traces": traces,
            "processing_time_ms": response.processing_time_ms
        }

    def _invoke_tool(self,
                    tool: ToolType,
                    args: Optional[Dict[str, Any]],
                    response: ImageResponse) -> Dict[str, Any]:
        """
        Invoke a tool based on the agent's decision.

        This is where the agent's actions happen.
        """
        if tool == ToolType.STORE_RESULT:
            return self._store_result(response, args)
        elif tool == ToolType.REQUEST_HUMAN_APPROVAL:
            return self._request_human_approval(response, args)
        elif tool == ToolType.SEND_ALERT:
            return self._send_alert(response, args)
        elif tool == ToolType.UPLOAD_TO_S3:
            return self._upload_to_s3(response, args)
        elif tool == ToolType.PROCESS_IMAGE:
            return {"status": "processing_started", "message": args.get("message", "")}
        elif tool == ToolType.RETRY_IMAGE:
            self.retry_count += 1
            if self.retry_count <= self.max_retries:
                return {"status": "retry_scheduled", "retry_number": self.retry_count}
            else:
                return {"status": "max_retries_exceeded"}
        else:
            return {"status": "unknown_tool", "tool": tool.value}

    def _store_result(self, response: ImageResponse, args: Optional[Dict]) -> Dict[str, Any]:
        """Store the processed result"""
        # In a real system, this would persist to a database
        return {
            "status": "stored",
            "detection_count": len(response.detections),
            "message": args.get("message") if args else "Results stored"
        }

    def _request_human_approval(self, response: ImageResponse, args: Optional[Dict]) -> Dict[str, Any]:
        """Request human approval for a decision"""
        approval_id = str(uuid.uuid4())
        self.pending_approvals[approval_id] = HumanApproval(
            requested_at=time.time(),
            requested_by="agent",
            message=args.get("message", "Please review detection results") if args else "Review required"
        )

        return {
            "status": "approval_requested",
            "approval_id": approval_id,
            "message": args.get("message", "Please review") if args else "Review requested"
        }

    def _send_alert(self, response: ImageResponse, args: Optional[Dict]) -> Dict[str, Any]:
        """Send an alert about the detection results"""
        # In a real system, this would send to SNS, Slack, etc.
        return {
            "status": "alert_sent",
            "message": args.get("message", "Alert sent") if args else "Alert triggered"
        }

    def _upload_to_s3(self, response: ImageResponse, args: Optional[Dict]) -> Dict[str, Any]:
        """Upload results to S3"""
        # In a real system, this would use boto3
        return {
            "status": "uploaded",
            "destination": args.get("bucket", "unknown") if args else "unknown_bucket"
        }

    def respond_to_approval(self, approval_id: str, decision: str, approver: str) -> Dict[str, Any]:
        """Record a human approval response"""
        if approval_id not in self.pending_approvals:
            return {"status": "approval_not_found"}

        approval = self.pending_approvals[approval_id]
        approval.decision = decision
        approval.response_at = time.time()
        approval.approver = approver

        return {
            "status": "approved" if decision == "approve" else "rejected",
            "approval_id": approval_id
        }

    def _store_trace(self, traces: List[Dict[str, Any]]) -> None:
        """Store traces for audit and analysis"""
        self.traces.extend(traces)

    def get_last_traces(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent trace records"""
        return self.traces[-count:]

    def clear_traces(self) -> None:
        """Clear stored traces"""
        self.traces = []