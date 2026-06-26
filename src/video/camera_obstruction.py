"""
camera_obstruction.py
Detects moments when the camera is being deliberately blocked or obstructed.
Consumes VideoAnalysisResult from frame_analysis.py — does NOT run YOLO again.

Detection strategies:
    1. Coverage — a person's bounding box covers too much of the frame area
    2. Darkness  — frame is suddenly much darker than baseline (hand over lens)
    3. Blur      — frame becomes significantly blurrier than baseline

All three are combined into ObstructionEvent objects with timestamps.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.models.video import (
    VideoAnalysisResult, FrameResult,
    ObstructionEvent, ObstructionResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class CameraObstructionDetector:
    """
    Detects camera obstruction using two complementary approaches:

    1. YOLO-based (from FrameResult):
       If a person's bounding box covers more than `coverage_threshold`
       of the frame area, it's likely someone is blocking the camera.

    2. Pixel-based (reads video frames directly):
       - Darkness: sudden drop in mean brightness vs baseline
       - Blur: sudden drop in Laplacian variance vs baseline

    Usage:
        from src.video.frame_analysis import analyze_frames
        from src.video.camera_obstruction import CameraObstructionDetector

        video_result = analyze_frames("video.mp4")
        detector = CameraObstructionDetector()
        result = detector.detect(video_result)

        for event in result.events:
            print(f"Obstruction at {event.start}s–{event.end}s ({event.reason})")
    """

    def __init__(
        self,
        coverage_threshold: float = 0.6,
        darkness_threshold: float = 0.4,
        blur_threshold: float = 0.2,
        min_event_duration: float = 0.5,
    ):
        """
        Args:
            coverage_threshold:  Person bbox must cover this fraction of frame
                                 area to count as obstruction (0..1).
            darkness_threshold:  Frame brightness must drop to this fraction
                                 of baseline to count as obstruction (0..1).
            blur_threshold:      Frame sharpness must drop to this fraction
                                 of baseline to count as obstruction (0..1).
            min_event_duration:  Ignore events shorter than this (seconds).
                                 Filters out single-frame false positives.
        """
        self.coverage_threshold = coverage_threshold
        self.darkness_threshold = darkness_threshold
        self.blur_threshold = blur_threshold
        self.min_event_duration = min_event_duration

        logger.info(
            "CameraObstructionDetector init (coverage=%.2f, darkness=%.2f, blur=%.2f)",
            coverage_threshold, darkness_threshold, blur_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, video_result: VideoAnalysisResult) -> ObstructionResult:
        """
        Detect all obstruction events in the video.

        Args:
            video_result: Output from FrameAnalyzer.process()

        Returns:
            ObstructionResult with list of ObstructionEvent.
        """
        events: list[ObstructionEvent] = []

        # Strategy 1: coverage check from YOLO detections (no video re-read)
        coverage_events = self._detect_by_coverage(video_result.frames)
        events.extend(coverage_events)

        # Strategy 2: pixel-based darkness + blur (reads video once)
        pixel_events = self._detect_by_pixels(video_result.file_path, video_result.fps)
        events.extend(pixel_events)

        # Merge overlapping events and filter short ones
        merged = self._merge_events(events)
        filtered = [e for e in merged if e.duration >= self.min_event_duration]

        result = ObstructionResult(
            file_path=video_result.file_path,
            events=sorted(filtered, key=lambda e: e.start),
        )

        logger.info(
            "Obstruction detection complete: %d events, %.1fs total obstructed",
            len(result.events), result.total_obstructed_seconds,
        )

        return result

    # ------------------------------------------------------------------
    # Strategy 1: YOLO coverage
    # ------------------------------------------------------------------

    def _detect_by_coverage(
        self,
        frames: list[FrameResult],
    ) -> list[ObstructionEvent]:
        """
        Flag frames where a person bbox covers too much of the frame.
        Uses normalized bbox coordinates from frame_analysis.py.
        """
        events: list[ObstructionEvent] = []
        in_event = False
        event_start = 0.0

        for frame in frames:
            max_coverage = max(
                (d.area for d in frame.persons),
                default=0.0,
            )
            is_obstructed = max_coverage >= self.coverage_threshold

            if is_obstructed and not in_event:
                in_event = True
                event_start = frame.timestamp
            elif not is_obstructed and in_event:
                in_event = False
                events.append(ObstructionEvent(
                    start=event_start,
                    end=frame.timestamp,
                    reason="coverage",
                    confidence=min(max_coverage / self.coverage_threshold, 1.0),
                ))

        # Close any open event at video end
        if in_event and frames:
            events.append(ObstructionEvent(
                start=event_start,
                end=frames[-1].timestamp,
                reason="coverage",
                confidence=1.0,
            ))

        logger.debug("Coverage-based obstructions: %d", len(events))
        return events

    # ------------------------------------------------------------------
    # Strategy 2: pixel-based darkness + blur
    # ------------------------------------------------------------------

    def _detect_by_pixels(
        self,
        file_path: str,
        fps: float,
    ) -> list[ObstructionEvent]:
        """
        Read video frames and detect sudden drops in brightness or sharpness.
        Computes baseline from the first 3 seconds of video.
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.warning("Cannot open video for pixel analysis: %s", file_path)
            return []

        baseline_brightness, baseline_blur = self._compute_baseline(cap, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

        events: list[ObstructionEvent] = []
        dark_start: Optional[float] = None
        blur_start: Optional[float] = None
        frame_index = 0
        sample_interval = max(1, int(fps))  # sample 1 fps

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % sample_interval == 0:
                    timestamp = frame_index / fps
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    brightness = float(np.mean(gray)) / 255.0
                    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                    # Darkness check
                    if baseline_brightness > 0:
                        brightness_ratio = brightness / baseline_brightness
                        if brightness_ratio < self.darkness_threshold:
                            if dark_start is None:
                                dark_start = timestamp
                        else:
                            if dark_start is not None:
                                events.append(ObstructionEvent(
                                    start=dark_start,
                                    end=timestamp,
                                    reason="darkness",
                                    confidence=1.0 - brightness_ratio,
                                ))
                                dark_start = None

                    # Blur check
                    if baseline_blur > 0:
                        blur_ratio = blur / baseline_blur
                        if blur_ratio < self.blur_threshold:
                            if blur_start is None:
                                blur_start = timestamp
                        else:
                            if blur_start is not None:
                                events.append(ObstructionEvent(
                                    start=blur_start,
                                    end=timestamp,
                                    reason="blur",
                                    confidence=1.0 - blur_ratio,
                                ))
                                blur_start = None

                frame_index += 1

            # Close any open events
            end_ts = frame_index / fps
            if dark_start is not None:
                events.append(ObstructionEvent(
                    start=dark_start, end=end_ts,
                    reason="darkness", confidence=1.0,
                ))
            if blur_start is not None:
                events.append(ObstructionEvent(
                    start=blur_start, end=end_ts,
                    reason="blur", confidence=1.0,
                ))
        finally:
            cap.release()

        logger.debug("Pixel-based obstructions: %d", len(events))
        return events

    @staticmethod
    def _compute_baseline(cap: cv2.VideoCapture, fps: float) -> tuple[float, float]:
        """Compute average brightness and blur from first 3 seconds."""
        baseline_frames = int(fps * 3)
        brightness_vals, blur_vals = [], []

        for _ in range(baseline_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_vals.append(float(np.mean(gray)) / 255.0)
            blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        avg_brightness = sum(brightness_vals) / len(brightness_vals) if brightness_vals else 1.0
        avg_blur = sum(blur_vals) / len(blur_vals) if blur_vals else 1.0

        logger.debug("Baseline — brightness: %.3f, blur: %.1f", avg_brightness, avg_blur)
        return avg_brightness, avg_blur

    # ------------------------------------------------------------------
    # Merge overlapping events
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_events(events: list[ObstructionEvent]) -> list[ObstructionEvent]:
        """Merge overlapping or adjacent events (within 1 second gap)."""
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.start)
        merged: list[ObstructionEvent] = [sorted_events[0]]

        for current in sorted_events[1:]:
            last = merged[-1]
            if current.start <= last.end + 1.0:
                # Extend the last event
                merged[-1] = ObstructionEvent(
                    start=last.start,
                    end=max(last.end, current.end),
                    reason=last.reason if last.reason == current.reason else "coverage",
                    confidence=max(last.confidence, current.confidence),
                )
            else:
                merged.append(current)

        return merged


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_obstructions(video_result: VideoAnalysisResult) -> ObstructionResult:
    """
    One-liner for use in analysis_pipeline.py.

    Example:
        from src.video.frame_analysis import analyze_frames
        from src.video.camera_obstruction import detect_obstructions

        video_result = analyze_frames("video.mp4")
        obstructions = detect_obstructions(video_result)

        if obstructions.has_obstructions:
            print(f"Camera blocked for {obstructions.total_obstructed_seconds}s")
    """
    return CameraObstructionDetector().detect(video_result)