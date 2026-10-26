#!/usr/bin/env python3
"""
VisionPilot Demo Script

Demonstrates the agentic vision loop with different input scenarios.
Shows how visual evidence changes agent decisions.
"""

import cv2
import numpy as np
import base64
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from visionpilot.core.pipeline import VisionPipeline
from visionpilot.agent.orchestrator import AgentOrchestrator


def create_test_image(brightness=128, objects=True, blur_kernel=None):
    """Create a test image with configurable properties"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * brightness

    if objects:
        # Add distinct objects
        cv2.rectangle(img, (50, 100), (150, 300), (255, 255, 255), -1)   # White rect
        cv2.rectangle(img, (450, 150), (550, 250), (0, 0, 0), -1)         # Black rect
        cv2.circle(img, (320, 240), 60, (200, 200, 200), -1)              # Gray circle
        cv2.rectangle(img, (200, 50), (280, 120), (255, 0, 0), -1)        # Blue rect
        cv2.circle(img, (500, 350), 40, (0, 255, 0), -1)                  # Green circle

    if blur_kernel:
        # Apply blur - kernel must be odd
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    success, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


def print_result(name, result):
    """Pretty print a result"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    print(f"  Outcome:      {result['outcome']}")
    print(f"  Decision:     {result['decision']['state']}")
    print(f"  Reason:       {result['decision']['reason']}")
    print(f"  Confidence:   {result['decision']['confidence']:.2f}")
    print(f"  Tool:         {result['decision']['tool']}")
    print(f"  Needs Approval: {result['decision']['needs_human_approval']}")
    print(f"  Quality:      {result['response']['validation']['quality']} ({result['response']['validation']['quality_score']:.2f})")
    print(f"  Detections:   {len(result['response']['detections'])}")
    for d in result['response']['detections']:
        print(f"    - {d['label']}: conf={d['confidence']:.2f}, box={d['bounding_box']}")
    print(f"  Processing:   {result['processing_time_ms']:.1f}ms")
    print(f"  Traces:       {len(result['traces'])} events")


def main():
    print("VISIONPILOT AGENTIC VISION DEMO")
    print("=" * 60)
    print("OpenCV Version:", cv2.__version__)

    # Initialize pipeline and orchestrator
    pipeline = VisionPipeline()
    orchestrator = AgentOrchestrator(pipeline)

    print("\n--- Running Demo Scenarios ---\n")

    # Scenario 1: Good quality image with objects
    print_result("HIGH QUALITY + OBJECTS", orchestrator.process_image(
        create_test_image(brightness=128, objects=True),
        'demo_good.jpg'
    ))

    # Scenario 2: Good quality, empty scene
    print_result("HIGH QUALITY + EMPTY", orchestrator.process_image(
        create_test_image(brightness=128, objects=False),
        'demo_empty.jpg'
    ))

    # Scenario 3: Blurry image
    print_result("BLURRY IMAGE", orchestrator.process_image(
        create_test_image(brightness=128, objects=True, blur_kernel=31),
        'demo_blur.jpg'
    ))

    # Scenario 4: Very dark image
    print_result("VERY DARK IMAGE", orchestrator.process_image(
        create_test_image(brightness=15, objects=True),
        'demo_dark.jpg'
    ))

    # Scenario 5: Medium quality with mixed detections
    print_result("MEDIUM QUALITY", orchestrator.process_image(
        create_test_image(brightness=100, objects=True),
        'demo_medium.jpg'
    ))

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Observations:")
    print("  1. Same agent, different inputs → different decisions")
    print("  2. Quality assessment drives rejection vs processing")
    print("  3. Detection confidence triggers human approval")
    print("  4. Structured traces provide full audit trail")
    print("\nTraces show the complete perception-decision-action loop!")


if __name__ == '__main__':
    main()