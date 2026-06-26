"""Data models for the video layer (frames, persons, obstruction, timeline)."""

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    """Types of timeline events. str-enum → сериализуется в JSON как строка."""
    PERSON_COUNT_CHANGE = "person_count_change"
    OBSTRUCTION = "obstruction"
    SPEAKER_CHANGE = "speaker_change"
    SILENCE = "silence"


@dataclass
class TimelineEvent:
    """A single event on the unified timeline."""
    timestamp: float            # когда произошло (seconds)
    event_type: EventType
    description: str            # человекочитаемо, попадёт в промпт LLM
    end: float | None = None    # для интервальных событий (obstruction, silence)
    data: dict = field(default_factory=dict)  # доп. структурированные поля

    @property
    def duration(self) -> float | None:
        return round(self.end - self.timestamp, 2) if self.end is not None else None

    def to_dict(self) -> dict:
        d = {
            "timestamp": round(self.timestamp, 2),
            "event_type": self.event_type.value,
            "description": self.description,
        }
        if self.end is not None:
            d["end"] = round(self.end, 2)
            d["duration"] = self.duration
        if self.data:
            d["data"] = self.data
        return d


@dataclass
class Timeline:
    """All events for one video, sorted by time."""
    file_path: str
    duration: float
    events: list["TimelineEvent"] = field(default_factory=list)

    def of_type(self, event_type: "EventType") -> list["TimelineEvent"]:
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "duration": round(self.duration, 2),
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

@dataclass
class Detection:
    """Single object detection in a frame. Bbox normalized 0..1."""
    label: str
    confidence: float
    x1: float
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
    """Detections for a single sampled video frame."""
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
# Person detection (statistics over frames)
# ---------------------------------------------------------------------------

@dataclass
class PersonDetectionResult:
    """Summary of people counts across the video."""
    max_count: int          # peak number of people — hint for diarization
    median_count: float
    avg_count: float
    peak_timestamp: float   # when the most people were visible (seconds)

    def to_dict(self) -> dict:
        return {
            "max_count": self.max_count,
            "median_count": self.median_count,
            "avg_count": round(self.avg_count, 2),
            "peak_timestamp": round(self.peak_timestamp, 2),
        }


# ---------------------------------------------------------------------------
# Camera obstruction
# ---------------------------------------------------------------------------

@dataclass
class ObstructionEvent:
    """A detected camera obstruction interval."""
    start: float
    end: float
    reason: str             # "coverage" | "darkness" | "blur"
    confidence: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": self.duration,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ObstructionResult:
    """All obstruction events detected in a video."""
    file_path: str
    events: list[ObstructionEvent] = field(default_factory=list)

    @property
    def total_obstructed_seconds(self) -> float:
        return round(sum(e.duration for e in self.events), 2)

    @property
    def has_obstructions(self) -> bool:
        return len(self.events) > 0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "has_obstructions": self.has_obstructions,
            "total_obstructed_seconds": self.total_obstructed_seconds,
            "events": [e.to_dict() for e in self.events],
        }