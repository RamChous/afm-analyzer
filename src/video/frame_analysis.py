"""
frame_analysis.py
Runs YOLOv8 once on a video and packages per-frame detection results.
All other video modules (person_detection, camera_obstruction, timeline_events)
consume FrameResult objects from here — YOLO is never loaded twice.

Flow:
    analyzer = FrameAnalyzer()
    frames = analyzer.process(video_path)   # runs YOLO once
    # pass frames to person_detection, camera_obstruction, etc.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single object detection in a frame."""
    label: str          # e.g. "person", "hand"
    confidence: float
    x1: float           # bounding box — normalized 0..1
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(self.x1, 4), round(self.y1, 4),
                     round(self.x2, 4), round(self.y2, 4)],
        }


@dataclass
class FrameResult:
    """Detections for a single video frame."""
    frame_index: int
    timestamp: float            # seconds from video start
    detections: list[Detection] = field(default_factory=list)

    @property
    def persons(self) -> list[Detection]:
        return [d for d in self.detections if d.label == "person"]

    @property
    def person_count(self) -> int:
        return len(self.persons)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "person_count": self.person_count,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class VideoAnalysisResult:
    """All frame results for one video."""
    file_path: str
    fps: float
    total_frames: int
    duration: float
    frames: list[FrameResult] = field(default_factory=list)

    def frames_with_persons(self) -> list[FrameResult]:
        return [f for f in self.frames if f.person_count > 0]

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": round(self.duration, 2),
            "frames_processed": len(self.frames),
        }


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class FrameAnalyzer:
    """
    Processes a video with YOLOv8 and returns per-frame detections.
    Runs locally — no external API calls.

    Usage:
        analyzer = FrameAnalyzer(model_size="n", sample_rate=1.0)
        result = analyzer.process("video.mp4")

        # pass to other modules
        from src.video.person_detection import PersonDetector
        speaker_count = PersonDetector().count(result)
    """

    # YOLO model sizes: n (nano) → s → m → l → x
    # For this task 'n' or 's' is fast enough and runs fine on CPU
    MODEL_SIZES = {"n", "s", "m", "l", "x"}

    def __init__(
        self,
        model_size: str = "n",
        confidence_threshold: float = 0.4,
        sample_rate: float = 1.0,
        classes: Optional[list[int]] = None,
    ):
        """
        Args:
            model_size:           YOLOv8 variant — 'n' is fastest, good for CPU.
            confidence_threshold: Minimum detection confidence to keep.
            sample_rate:          Frames per second to sample (e.g. 1.0 = 1 fps).
                                  Lower = faster processing, less detail.
            classes:              YOLO class IDs to detect. None = all classes.
                                  Person = 0. Pass [0] to detect only people.
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"model_size must be one of {self.MODEL_SIZES}")

        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.classes = classes
        self._model: Optional[YOLO] = None

        logger.info(
            "FrameAnalyzer init (model=yolov8%s, conf=%.2f, sample_rate=%.1f fps)",
            model_size, confidence_threshold, sample_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str) -> VideoAnalysisResult:
        """
        Run YOLO on the video and return all frame detections.

        Args:
            file_path: Path to video file.

        Returns:
            VideoAnalysisResult containing FrameResult for each sampled frame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # How many frames to skip between samples
        frame_interval = max(1, int(fps / self.sample_rate))

        logger.info(
            "Processing video: %s (%.1fs, %.0f fps, sampling every %d frames)",
            path.name, duration, fps, frame_interval,
        )

        model = self._load_model()
        frame_results: list[FrameResult] = []
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_interval == 0:
                    timestamp = frame_index / fps
                    detections = self._detect(model, frame)
                    frame_results.append(FrameResult(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        detections=detections,
                    ))

                frame_index += 1
        finally:
            cap.release()

        logger.info(
            "Frame analysis complete: %d frames sampled, max persons in frame: %d",
            len(frame_results),
            max((f.person_count for f in frame_results), default=0),
        )

        return VideoAnalysisResult(
            file_path=str(path),
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            frames=frame_results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> YOLO:
        """Lazy-load YOLOv8 model (cached after first call)."""
        if self._model is None:
            model_name = f"yolov8{self.model_size}.pt"
            logger.info("Loading YOLO model: %s", model_name)
            # ultralytics auto-downloads on first use, then caches locally
            self._model = YOLO(model_name)
            logger.info("YOLO model loaded.")
        return self._model

    def _detect(self, model: YOLO, frame) -> list[Detection]:
        """Run YOLO on a single frame and return normalized detections."""
        h, w = frame.shape[:2]
        results = model(
            frame,
            conf=self.confidence_threshold,
            classes=self.classes,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append(Detection(
                    label=label,
                    confidence=confidence,
                    # normalize to 0..1 for resolution-independence
                    x1=x1 / w, y1=y1 / h,
                    x2=x2 / w, y2=y2 / h,
                ))

        return detections


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_frames(
    file_path: str,
    model_size: str = "n",
    sample_rate: float = 1.0,
) -> VideoAnalysisResult:
    """
    One-liner for use in analysis_pipeline.py.

    Example:
        from src.video.frame_analysis import analyze_frames

        video_result = analyze_frames("video.mp4", sample_rate=1.0)
        # then pass to person_detection, camera_obstruction, timeline_events
    """
    return FrameAnalyzer(
        model_size=model_size,
        sample_rate=sample_rate,
        classes=[0],  # 0 = person only, speeds up processing
    ).process(file_path)