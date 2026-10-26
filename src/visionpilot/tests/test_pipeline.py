"""
Unit tests for the vision pipeline.
"""

import pytest
import cv2
import numpy as np
import base64
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from visionpilot.core.pipeline import (
    VisionPipeline, ImageRequest, ImageResponse,
    ImageQuality, DetectionResult, MeasurementResult
)


class TestVisionPipeline:
    """Tests for the VisionPipeline class"""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing"""
        return VisionPipeline()

    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(img, (50, 100), (150, 300), (255, 255, 255), -1)
        cv2.rectangle(img, (450, 150), (550, 250), (0, 0, 0), -1)
        cv2.circle(img, (320, 240), 60, (200, 200, 200), -1)

        success, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()

    def test_opencv_version(self, pipeline):
        """Test that OpenCV 5 is being used"""
        import cv2
        version = tuple(map(int, cv2.__version__.split('.')[:2]))
        assert version >= (5, 0), f"OpenCV version should be 5.x, got {cv2.__version__}"

    def test_validate_image_valid(self, pipeline, sample_image_bytes):
        """Test validation of a valid image"""
        result = pipeline.validate_image(sample_image_bytes)

        assert result.is_valid is True
        assert result.width == 640
        assert result.height == 480
        assert result.channels == 3
        assert result.quality_score > 0

    def test_validate_image_invalid(self, pipeline):
        """Test validation of an invalid image"""
        result = pipeline.validate_image(b"invalid image data")

        assert result.is_valid is False
        assert result.quality == ImageQuality.INVALID

    def test_validate_image_quality_too_small(self, pipeline):
        """Test validation rejects tiny images"""
        # Create a tiny image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        success, buffer = cv2.imencode('.jpg', img)
        result = pipeline.validate_image(buffer.tobytes())

        assert result.is_valid is False

    def test_detect_objects_finds_contours(self, pipeline, sample_image_bytes):
        """Test that object detection finds contours"""
        np_array = np.frombuffer(sample_image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        detections = pipeline.detect_objects(image)

        assert len(detections) > 0
        for det in detections:
            assert isinstance(det, DetectionResult)
            assert det.confidence > 0
            assert det.bounding_box[2] > 0 and det.bounding_box[3] > 0  # w, h > 0

    def test_process_request_complete(self, pipeline, sample_image_bytes):
        """Test full processing pipeline"""
        request = ImageRequest(
            request_id='test-123',
            image_data=sample_image_bytes,
            filename='test.jpg'
        )

        response = pipeline.process_request(request)

        assert isinstance(response, ImageResponse)
        assert response.request_id == 'test-123'
        assert response.validation.is_valid is True
        assert len(response.detections) > 0
        assert len(response.trace_events) > 0
        assert response.processing_time_ms > 0

    def test_create_annotated_image(self, pipeline, sample_image_bytes):
        """Test that annotated images are created"""
        np_array = np.frombuffer(sample_image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        detections = pipeline.detect_objects(image)
        annotated = pipeline.create_annotated_image(image, detections)

        assert annotated is not None
        assert annotated.shape == image.shape

    def test_measure_distance(self, pipeline):
        """Test distance measurement between detections"""
        det1 = DetectionResult(
            bounding_box=(0, 0, 100, 100),
            confidence=0.9,
            label="test",
            center=(50.0, 50.0),
            area_pixels=10000,
            normalized_center=(0.5, 0.5)
        )
        det2 = DetectionResult(
            bounding_box=(100, 100, 100, 100),
            confidence=0.9,
            label="test",
            center=(150.0, 150.0),
            area_pixels=10000,
            normalized_center=(0.5, 0.5)
        )

        measurement = pipeline.measure_distance(det1, det2)

        # Distance should be sqrt((100)^2 + (100)^2) ≈ 141.4
        assert measurement.pixel_distance > 100
        assert measurement.pixel_distance < 200

    def test_blurry_image_detection(self, pipeline):
        """Test that blurry images are detected"""
        # Create an image with some detail, then blur it heavily
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some patterns
        cv2.rectangle(img, (100, 100), (200, 300), (255, 255, 255), -1)
        cv2.circle(img, (400, 200), 80, (128, 128, 128), -1)

        # Apply very heavy blur multiple times
        img = cv2.GaussianBlur(img, (51, 51), 0)
        img = cv2.GaussianBlur(img, (51, 51), 0)

        success, buffer = cv2.imencode('.jpg', img)
        result = pipeline.validate_image(buffer.tobytes())

        assert result.quality in [ImageQuality.BLURRY, ImageQuality.POOR]
        assert result.quality_score < 0.5


class TestAgenticVision:
    """Tests for the agentic vision loop"""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance"""
        from visionpilot.agent.orchestrator import AgentOrchestrator
        return AgentOrchestrator()

    def test_agent_makes_different_decisions_for_different_images(self):
        """Test that visual evidence changes agent decisions"""
        from visionpilot.agent.orchestrator import AgentOrchestrator
        from visionpilot.core.pipeline import VisionPipeline

        pipeline = VisionPipeline()
        orchestrator = AgentOrchestrator(pipeline)

        # Create a good quality image
        good_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(good_img, (100, 100), (200, 300), (255, 255, 255), -1)
        success, buffer = cv2.imencode('.jpg', good_img)
        good_bytes = buffer.tobytes()

        # Create a very dark image
        dark_img = np.ones((480, 640, 3), dtype=np.uint8) * 10
        success, buffer = cv2.imencode('.jpg', dark_img)
        dark_bytes = buffer.tobytes()

        # Process good image
        good_result = orchestrator.process_image(good_bytes, 'good.jpg')

        # Process dark image
        dark_result = orchestrator.process_image(dark_bytes, 'dark.jpg')

        # Decisions should differ
        assert good_result['outcome'] != dark_result['outcome'] or \
               good_result['decision']['state'] != dark_result['decision']['state']

        # Good image should succeed
        assert good_result['outcome'] in ['success', 'completed']

    def test_trace_records_are_created(self):
        """Test that structured traces are recorded"""
        from visionpilot.agent.orchestrator import AgentOrchestrator
        from visionpilot.core.pipeline import VisionPipeline

        pipeline = VisionPipeline()
        orchestrator = AgentOrchestrator(pipeline)

        # Create test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(img, (100, 100), (200, 300), (255, 255, 255), -1)
        success, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()

        result = orchestrator.process_image(img_bytes, 'test.jpg')

        # Check traces exist
        traces = result['traces']
        assert len(traces) > 0

        # Check trace structure
        # At least one trace should have final_outcome, and one should have decision
        has_final_outcome = any('final_outcome' in t for t in traces)
        has_decision = any('decision' in t for t in traces)

        for trace in traces:
            assert 'request_id' in trace
            assert 'timestamp' in trace

        assert has_final_outcome, "No trace has final_outcome"
        assert has_decision, "No trace has decision"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])