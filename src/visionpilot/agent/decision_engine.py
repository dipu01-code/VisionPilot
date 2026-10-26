"""
Decision engine for the agentic vision system.

Implements deterministic rules for:
- Quality threshold checks
- Detection result interpretation
- Approval requirements
- Retry logic
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import logging

from ..core.pipeline import ImageValidationResult, DetectionResult, ImageResponse
from .types import AgentDecision, AgentState, ToolType

logger = logging.getLogger(__name__)


@dataclass
class DecisionRule:
    """A rule that produces a decision based on visual evidence"""
    name: str
    priority: int  # Higher priority rules fire first
    condition: Callable[[ImageResponse], bool]
    action: Callable[[ImageResponse], AgentDecision]
    description: str = ""

    def __lt__(self, other):
        return self.priority < other.priority


class DecisionEngine:
    """
    Deterministic decision engine that produces actions
    based on OpenCV 5 analysis results.

    Rules are evaluated in priority order (highest first).
    """

    # Confidence thresholds
    GRAY_TOLERANCE = 0.3      # Don't trust low confidence
    HIGH_CONFIDENCE = 0.7     # High confidence threshold
    APPROVAL_CONFIDENCE = 0.9 # Require human approval below this

    # Quantity thresholds
    MIN_DETECTIONS_FOR_ANALYSIS = 1
    DETECTIONS_PER_PERSON_THRESHOLD = 3  # Multiple people indicate crowded scene

    def __init__(self):
        self.rules = self._build_rules()

    def _build_rules(self) -> List[DecisionRule]:
        """Build the ordered list of decision rules"""
        rules = [
            # Rule: Reject low quality images
            DecisionRule(
                name="reject_low_quality",
                priority=100,
                condition=lambda r: not r.validation.is_valid,
                action=self._reject_low_quality,
                description="Reject images that fail validation"
            ),
            DecisionRule(
                name="reject_blurry",
                priority=95,
                condition=lambda r: r.validation.quality_score < 0.3,
                action=self._reject_blurry,
                description="Reject blurry images"
            ),
            # Rule: Insufficient detections for meaningful analysis
            DecisionRule(
                name="no_detections",
                priority=90,
                condition=lambda r: len(r.detections) == 0,
                action=self._no_detections_action,
                description="Handle images with no detections"
            ),
            # Rule: Request approval for ambiguous cases
            DecisionRule(
                name="ambiguous_detections_low_confidence",
                priority=80,
                condition=lambda r: len(r.detections) > 0 and
                                   max(d.confidence for d in r.detections) < self.APPROVAL_CONFIDENCE,
                action=self._ask_approval,
                description="Request human approval for low-confidence detections"
            ),
            # Rule: Good detection results
            DecisionRule(
                name="good_detections",
                priority=70,
                condition=lambda r: (len(r.detections) >= self.MIN_DETECTIONS_FOR_ANALYSIS and
                                    all(d.confidence > self.GRAY_TOLERANCE for d in r.detections)),
                action=self._good_detections,
                description="Accept image with good detections"
            ),
            # Rule: Mixed quality detections
            DecisionRule(
                name="mixed_quality",
                priority=60,
                condition=lambda r: len(r.detections) > 0 and
                                   any(d.confidence <= self.GRAY_TOLERANCE for d in r.detections),
                action=self._mixed_quality,
                description="Handle mixed quality detections"
            ),
            # Rule: High object density (crowded scene)
            DecisionRule(
                name="crowded_scene",
                priority=50,
                condition=lambda r: len(r.detections) > self.DETECTIONS_PER_PERSON_THRESHOLD,
                action=self._crowded_scene,
                description="Handle crowded scenes with multiple detections"
            ),
        ]

        # Sort by priority (highest first)
        return sorted(rules, reverse=True)

    def decide(self, response: ImageResponse) -> AgentDecision:
        """
        Produce a decision based on image analysis results.

        Evaluates rules in priority order and returns the first matching decision.
        """
        for rule in self.rules:
            try:
                if rule.condition(response):
                    logger.debug(f"Rule matched: {rule.name}")
                    return rule.action(response)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
                continue

        # Default: retry with better conditions
        return AgentDecision(
            decision_id="",
            state=AgentState.RESTARTED,
            reason="No rules matched - defaulting to retry",
            confidence=0.5,
            tool=ToolType.RETRY_IMAGE,
            tool_args={"reason": "rule_not_matched"}
        )

    def _reject_low_quality(self, response: ImageResponse) -> AgentDecision:
        """Reject image that fails validation"""
        return AgentDecision(
            decision_id="",
            state=AgentState.REJECTED,
            reason=f"Image failed validation: {response.validation.error_message}",
            confidence=0.95,
            tool=ToolType.REQUEST_HUMAN_APPROVAL,
            tool_args={"message": f"Low quality image rejected. Quality: {response.validation.quality.value}"},
            needs_human_approval=False  # No approval needed for rejection
        )

    def _reject_blurry(self, response: ImageResponse) -> AgentDecision:
        """Reject blurry image"""
        return AgentDecision(
            decision_id="",
            state=AgentState.REJECTED,
            reason=f"Image too blurry (score: {response.validation.quality_score:.2f})",
            confidence=0.9,
            tool=ToolType.REQUEST_HUMAN_APPROVAL,
            tool_args={"message": "Image rejected due to blur"},
            needs_human_approval=False
        )

    def _no_detections_action(self, response: ImageResponse) -> AgentDecision:
        """Handle image with no object detections"""
        # Check if this is a scenario requiring a different approach
        if response.validation.quality_score > 0.7:
            # Good quality but no detections - could be empty scene
            return AgentDecision(
                decision_id="",
                state=AgentState.COMPLETED,
                reason="Good quality image with no objects detected - scene appears empty",
                confidence=0.8,
                tool=None
            )
        else:
            # Could use another image
            return AgentDecision(
                decision_id="",
                state=AgentState.NEEDS_APPROVAL,
                reason="No objects detected in potentially suboptimal image",
                confidence=0.6,
                tool=ToolType.REQUEST_HUMAN_APPROVAL,
                tool_args={"message": "No objects detected. Consider using a better image or different focus."},
                needs_human_approval=True
            )

    def _ask_approval(self, response: ImageResponse) -> AgentDecision:
        """Request human approval for low-confidence detections"""
        low_conf_detections = [d for d in response.detections if d.confidence <= self.APPROVAL_CONFIDENCE]

        return AgentDecision(
            decision_id="",
            state=AgentState.NEEDS_APPROVAL,
            reason=f"Detections have low confidence: {[d.label for d in low_conf_detections]}",
            confidence=0.5,
            tool=ToolType.REQUEST_HUMAN_APPROVAL,
            tool_args={
                "message": "Low confidence detections detected. Please review.",
                "detections": [{"label": d.label, "confidence": d.confidence} for d in low_conf_detections]
            },
            needs_human_approval=True
        )

    def _good_detections(self, response: ImageResponse) -> AgentDecision:
        """Accept image with good detections"""
        total_confidence = sum(d.confidence for d in response.detections) / len(response.detections)
        avg_area = sum(d.area_pixels for d in response.detections) / len(response.detections)

        return AgentDecision(
            decision_id="",
            state=AgentState.COMPLETED,
            reason=f"Good detections found. Avg confidence: {total_confidence:.2f}",
            confidence=total_confidence,
            tool=ToolType.STORE_RESULT,
            tool_args={
                "message": "Image accepted",
                "detection_count": len(response.detections),
                "avg_confidence": total_confidence,
                "avg_area": avg_area
            }
        )

    def _mixed_quality(self, response: ImageResponse) -> AgentDecision:
        """Handle mixed quality detections"""
        high_conf = [d for d in response.detections if d.confidence > self.HIGH_CONFIDENCE]
        low_conf = [d for d in response.detections if d.confidence <= self.HIGH_CONFIDENCE]

        if len(high_conf) > 0:
            # Some detections are good - keep them
            return AgentDecision(
                decision_id="",
                state=AgentState.COMPLETED,
                reason=f"Mixed quality: {len(high_conf)} high-confidence, {len(low_conf)} low-confidence",
                confidence=0.7,
                tool=ToolType.STORE_RESULT,
                tool_args={
                    "message": "Mixed quality detections - high confidence results accepted",
                    "high_confidence_count": len(high_conf),
                    "low_confidence_count": len(low_conf)
                },
                needs_human_approval=len(low_conf) > 2
            )
        else:
            # No high confidence detections
            return AgentDecision(
                decision_id="",
                state=AgentState.NEEDS_APPROVAL,
                reason="All detections are low confidence",
                confidence=0.5,
                tool=ToolType.REQUEST_HUMAN_APPROVAL,
                tool_args={"message": "All detections have low confidence. Please review."},
                needs_human_approval=True
            )

    def _crowded_scene(self, response: ImageResponse) -> AgentDecision:
        """Handle crowded scenes with many detections"""
        return AgentDecision(
            decision_id="",
            state=AgentState.COMPLETED,
            reason=f"Crowded scene detected with {len(response.detections)} objects",
            confidence=0.85,
            tool=ToolType.PROCESS_IMAGE,
            tool_args={
                "message": "Crowded scene - full analysis performed",
                "object_count": len(response.detections),
                "density": len(response.detections) / (response.validation.width * response.validation.height)
            }
        )

    def get_decision_trace(self, response: ImageResponse) -> Dict[str, Any]:
        """
        Get a detailed trace of why a decision was made.

        Useful for debugging and understanding agent behavior.
        """
        decision = self.decide(response)

        return {
            "decision_rules_evaluated": [r.name for r in self.rules],
            "selected_rule": decision.state.value if hasattr(decision, 'state') else "unknown",
            "confidence_indicators": {
                "quality_score": response.validation.quality_score,
                "detection_count": len(response.detections),
                "avg_detection_confidence": sum(d.confidence for d in response.detections) / len(response.detections) if response.detections else 0,
                "max_detection_confidence": max((d.confidence for d in response.detections), default=0)
            },
            "decision": {
                "state": decision.state.value,
                "reason": decision.reason,
                "needs_approval": decision.needs_human_approval,
                "tool": decision.tool.value if decision.tool else None
            }
        }