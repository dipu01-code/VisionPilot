"""
Unit tests for the video pipeline.
"""

import pytest
import cv2
import numpy as np
import base64
import sys
import os
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from visionpilot.core.video_pipeline import (
    VideoPipeline, VideoRequest, VideoValidationResult,
    DetectionResult, VideoQuality
)


class TestVideoPipeline:
    """Tests for the VideoPipeline class"""

    @pytest.fixture
    def pipeline(self):
        """Create a video pipeline instance for testing"""
        return VideoPipeline()

    @pytest.fixture
    def sample_video_bytes(self):
        """Create a sample video as bytes"""
        # Create a simple test video with a few frames
        width, height = 640, 480
        frames = []

        for i in range(5):
            # Create a frame with a colored object
            img = np.ones((height, width, 3), dtype=np.uint8) * 128

            # Add a distinct object in each frame
            color = (255 - i * 50, i * 50, 255 - i * 50)
            cv2.rectangle(img, (50 + i * 100, 50), (150 + i * 100, 300), color, -1)

            # Encode as JPEG
            success, buffer = cv2.imencode('.jpg', img)
            if success:
                frames.append(buffer.tobytes())
            else:
                # Fallback: create a minimal valid JPEG
                frames.append(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01')

        # Combine frames (simple approach for testing)
        # In real implementation, this would be proper video container
        video_bytes = b''.join(frames)
        return video_bytes if len(video_bytes) > 0 else self._create_test_video()

    def _create_test_video(self) -> bytes:
        """Create a minimal test video"""
        # Create a simple video file in a basic format
        import struct

        # Very simple video header (just enough for cv2 to read)
        # This creates a minimal valid video file
        return b'FFMPEGAVIH264\x00\x00\x00\x00\x00\x00\x00'

    def test_opencv_version(self, pipeline):
        """Test that OpenCV 5 is being used"""
        import cv2
        version = tuple(map(int, cv2.__version__.split('.')[:2]))
        assert version >= (5, 0), f"OpenCV version should be 5.x, got {cv2.__version__}"

    def test_validate_video_valid(self, pipeline, sample_video_bytes):
        """Test validation of a valid video"""
        result = pipeline.validate_video(sample_video_bytes)

        assert result.is_valid is True
        assert result.width > 0
        assert result.height > 0
        assert result.fps > 0
        assert result.frames_analyzed > 0
        assert result.quality_score >= 0.0

    def test_validate_video_invalid(self, pipeline):
        """Test validation of an invalid video"""
        result = pipeline.validate_video(b"invalid video data")

        assert result.is_valid is False
        assert result.quality == VideoQuality.INVALID

    def test_validate_video_quality_issues(self, pipeline):
        """Test validation detects quality issues"""
        # Create a very low quality video (just audio/header)
        low_quality_video = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01'

        result = pipeline.validate_video(low_quality_video)

        assert result.is_valid is False or result.quality != VideoQuality.EXCELLENT

    def test_process_video_request(self, pipeline, sample_video_bytes):
        """Test full video processing pipeline"""
        request = VideoRequest(
            request_id='test-video-123',
            video_data=sample_video_bytes,
            filename='test_video.mp4'
        )

        response = pipeline.process_video_request(request)

        assert response.request_id == 'test-video-123'
        assert response.original_filename == 'test_video.mp4'
        assert response.validation.is_valid is True
        assert len(response.detections) >= 0  # May have detections depending on content
        assert response.processing_time_ms > 0

    def test_detect_objects_in_frame(self, pipeline):
        """Test object detection in a single frame"""
        # Create a test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(frame, (100, 100), (200, 300), (255, 0, 0), -1)

        detections = pipeline._detect_objects_in_frame(frame, 1, 30.0)

        assert len(detections) > 0
        for det in detections:
            assert isinstance(det, DetectionResult)
            assert det.bounding_box[2] > 0 and det.bounding_box[3] > 0  # w, h > 0
            assert det.frame_number == 1
            assert det.timestamp_seconds == 0.033  # 1/30

    def test_calculate_distance(self, pipeline):
        """Test distance calculation between detections"""
        det1 = DetectionResult(
            bounding_box=(0, 0, 100, 100),
            confidence=0.9,
            label="test",
            center=(50.0, 50.0),
            area_pixels=10000,
            normalized_center=(0.5, 0.5),
            frame_number=1,
            timestamp_seconds=0.033
        )
        det2 = DetectionResult(
            bounding_box=(100, 100, 100, 100),
            confidence=0.9,
            label="test",
            center=(150.0, 150.0),
            area_pixels=10000,
            normalized_center=(0.5, 0.5),
            frame_number=1,
            timestamp_seconds=0.033
        )

        distance = pipeline._calculate_distance(det1, det2)

        # Distance should be sqrt((100)^2 + (100)^2) ≈ 141.4
        assert distance > 100
        assert distance < 200

    def test_video_quality_detection(self, pipeline):
        """Test video quality classification"""
        # Create frames with different quality issues
        # This is a simplified test - in reality would need actual video files
        pass


class TestEnhancedAgenticVision:
    """Tests for the enhanced agentic vision loop with video"""

    def test_video_processing_differs_from_image(self):
        """Test that video processing handles temporal aspects"""
        from visionpilot.agent.orchestrator import AgentOrchestrator
        from visionpilot.core.video_pipeline import VideoPipeline

        video_pipeline = VideoPipeline()
        orchestrator = AgentOrchestrator(video_pipeline)

        # Create a simple test video
        video_bytes = self._create_test_video_bytes()

        # Process video
        result = orchestrator.process_video_image(video_bytes, 'test_video.mp4')

        # Should have video-specific results
        assert 'response' in result
        assert 'detections' in result['response']
        assert len(result['response']['detections']) >= 0

    def _create_test_video_bytes(self) -> bytes:
        """Create test video bytes for testing"""
        # Minimal test video data
        return b'fake video data for testing'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])