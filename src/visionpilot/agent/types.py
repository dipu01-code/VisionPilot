"""
Core types for the agentic vision system.

Defines the agent state machine, decisions, and trace records.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import uuid
import time


class AgentState(Enum):
    """Finite states of the agent orchestrator"""
    # Initial state
    IDLE = "idle"

    # Processing states
    RECEIVED = "received"          # Request received
    VALIDATING = "validating"      # Validating input
    ANALYZING = "analyzing"        # Running OpenCV analysis
    INTERPRETING = "interpreting"  # Interpreting results

    # Decision states
    DECIDING = "deciding"
    NEEDS_APPROVAL = "needs_approval"

    # Action states
    PROCESSING = "processing"
    WAITING = "waiting"

    # Terminal states
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    RESTARTED = "restarted"


class ToolType(Enum):
    """Available tools the agent can invoke"""
    # Processing tools
    PROCESS_IMAGE = "process_image"
    RETRY_IMAGE = "retry_image"
    LOAD_REFERENCE_IMAGE = "load_reference_image"

    # Decision tools
    REQUEST_HUMAN_APPROVAL = "request_human_approval"
    SEND_ALERT = "send_alert"
    REQUEST_ADDITIONAL_DATA = "request_additional_data"

    # Storage tools
    STORE_RESULT = "store_result"
    UPLOAD_TO_S3 = "upload_to_s3"


@dataclass
class AgentDecision:
    """A decision made by the agent"""
    decision_id: str
    state: AgentState
    reason: str
    confidence: float  # 0.0 to 1.0
    tool: Optional[ToolType] = None
    tool_args: Optional[Dict[str, Any]] = None
    needs_human_approval: bool = False

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = str(uuid.uuid4())


@dataclass
class AgentTrace:
    """
    Structured trace record for the agentic vision loop.

    Captures the decision path for auditability and debugging.
    """
    trace_id: str
    request_id: str
    observation_id: str
    timestamp: float
    opencv_operation: Optional[str] = None
    visual_findings: Optional[Dict[str, Any]] = None
    confidence_indicators: Optional[Dict[str, float]] = None
    decision: Optional[AgentDecision] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    retry_decision: Optional[str] = None
    human_approval: Optional[Dict[str, Any]] = None
    final_outcome: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
        }

        if self.opencv_operation:
            result["opencv_operation"] = self.opencv_operation

        if self.visual_findings:
            result["visual_findings"] = self.visual_findings

        if self.confidence_indicators:
            result["confidence_indicators"] = self.confidence_indicators

        if self.decision:
            result["decision"] = {
                "decision_id": self.decision.decision_id,
                "state": self.decision.state.value,
                "reason": self.decision.reason,
                "confidence": self.decision.confidence,
                "tool": self.decision.tool.value if self.decision.tool else None,
                "needs_human_approval": self.decision.needs_human_approval,
            }

        if self.tool_arguments:
            result["tool_arguments"] = self.tool_arguments

        if self.tool_result:
            result["tool_result"] = self.tool_result

        if self.retry_decision:
            result["retry_decision"] = self.retry_decision

        if self.human_approval:
            result["human_approval"] = self.human_approval

        if self.final_outcome:
            result["final_outcome"] = self.final_outcome

        return result