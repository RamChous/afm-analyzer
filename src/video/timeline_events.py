"""
person_detection.py
Counts people from pre-computed VideoAnalysisResult (frame_analysis.py).
Does NOT run YOLO — consumes ready FrameResult objects.

Main output: speaker_count → passed to diarization.py
"""

import logging
from dataclasses import dataclass
from statistics import median

from src.video.frame_analysis import VideoAnalysisResult, FrameResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PersonDetectionResult:
    """Summary of people counts across the video."""
    max_count: int          # peak number of people — used as speaker_count
    median_count: float     # median across all frames
    avg_count: float        # average across all frames
    peak_timestamp: float   # when the most people were visible (seconds)

    def to_dict(self) -> dict:
        return {
            "max_count": self.max_count,
            "median_count": self.median_count,
            "avg_count": round(self.avg_count, 2),
            "peak_timestamp": round(self.peak_timestamp, 2),
        }


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class PersonDetector:
    """
    Analyses VideoAnalysisResult and returns person count statistics.

    Usage:
        from src.video.frame_analysis import analyze_frames
        from src.video.person_detection import PersonDetector

        video_result = analyze_frames("video.mp4")
        result = PersonDetector().count(video_result)

        print(result.max_count)   # → speaker_count for diarization
    """

    def count(self, video_result: VideoAnalysisResult) -> PersonDetectionResult:
        """
        Compute person count statistics from frame detections.

        Args:
            video_result: Output from FrameAnalyzer.process()

        Returns:
            PersonDetectionResult with max/median/avg counts and peak timestamp.
        """
        frames = video_result.frames

        if not frames:
            logger.warning("No frames to analyse in %s", video_result.file_path)
            return PersonDetectionResult(
                max_count=0,
                median_count=0.0,
                avg_count=0.0,
                peak_timestamp=0.0,
            )

        counts = [f.person_count for f in frames]

        max_count = max(counts)
        peak_frame = next(f for f in frames if f.person_count == max_count)

        result = PersonDetectionResult(
            max_count=max_count,
            median_count=median(counts),
            avg_count=sum(counts) / len(counts),
            peak_timestamp=peak_frame.timestamp,
        )

        logger.info(
            "Person detection: max=%d, median=%.1f, avg=%.2f, peak at %.1fs",
            result.max_count, result.median_count,
            result.avg_count, result.peak_timestamp,
        )

        return result

    def count_over_time(
        self,
        video_result: VideoAnalysisResult,
    ) -> list[tuple[float, int]]:
        """
        Returns person count at each sampled timestamp.
        Useful for timeline_events.py.

        Returns:
            List of (timestamp, person_count) tuples.
        """
        return [(f.timestamp, f.person_count) for f in video_result.frames]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_persons(video_result: VideoAnalysisResult) -> PersonDetectionResult:
    """
    One-liner for use in analysis_pipeline.py.

    Example:
        from src.video.frame_analysis import analyze_frames
        from src.video.person_detection import detect_persons
        from src.speech.diarization import diarize_video

        video_result   = analyze_frames("video.mp4")
        persons        = detect_persons(video_result)
        diarization    = diarize_video(
            file_path="video.mp4",
            transcription=transcription,
            hf_token="hf_...",
            speaker_count=persons.max_count,
        )
    """
    return PersonDetector().count(video_result)