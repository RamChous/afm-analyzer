"""
frame_analysis.py
Runs YOLOv8 once on a video and packages per-frame detection results.
All other video modules (person_detection, camera_obstruction, timeline_events)
consume FrameResult objects from here — YOLO is never loaded twice.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO
from src.models.video import Detection, FrameResult, VideoAnalysisResult

logger = logging.getLogger(__name__)

class FrameAnalyzer:
    """
    Processes a video with YOLOv8 and returns per-frame detections.
    Runs locally — no external API calls.

    Usage:
        analyzer = FrameAnalyzer(sample_rate=1.0)
        result = analyzer.process("video.mp4")
    """

    MODEL_SIZES = {"n", "s", "m", "l", "x"}

    def __init__(
        self,
        model_size: str = "n",
        confidence_threshold: float = 0.4,
        sample_rate: float = 3.0,
        classes: Optional[list[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_size:           YOLOv8 variant — 'n' is fastest, good for CPU.
            confidence_threshold: Minimum detection confidence to keep.
            sample_rate:          Frames per second to sample (e.g. 1.0 = 1 fps).
            classes:              YOLO class IDs to detect. None = all.
                                  Person = 0. Pass [0] to detect only people.
            device:               'cpu', 'cuda', or None (ultralytics auto-picks).
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"model_size must be one of {self.MODEL_SIZES}")

        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.classes = classes
        self.device = device
        self._model: Optional[YOLO] = None

        logger.info(
            "FrameAnalyzer init (model=yolov8%s, conf=%.2f, sample_rate=%.1f fps)",
            model_size, confidence_threshold, sample_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str) -> VideoAnalysisResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        frame_interval = max(1, int(fps / self.sample_rate))

        logger.info(
            "Processing video: %s (%.1fs, %.0f fps, sampling every %d frames)",
            path.name, duration, fps, frame_interval,
        )

        model = self._load_model()
        frame_results: list[FrameResult] = []
        frame_index = 0
        # Сколько кадров мы реально проанализируем — для прогресса
        expected_samples = max(1, total_frames // frame_interval)

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
        if self._model is None:
            model_name = f"yolov8{self.model_size}.pt"
            logger.info("Loading YOLO model: %s", model_name)
            self._model = YOLO(model_name)
            logger.info("YOLO model loaded.")
        return self._model

    def _detect(self, model: YOLO, frame) -> list[Detection]:
        h, w = frame.shape[:2]
        results = model(
            frame,
            conf=self.confidence_threshold,
            classes=self.classes,
            device=self.device,
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
    sample_rate: float = 3.0,
) -> VideoAnalysisResult:
    return FrameAnalyzer(
        model_size=model_size,
        sample_rate=sample_rate,
        classes=[0],  # 0 = person only, speeds up processing
    ).process(file_path)