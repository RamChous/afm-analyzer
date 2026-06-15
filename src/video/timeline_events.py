"""
timeline_events.py
Aggregates all detected events (persons, obstructions, speech) into a single
chronological timeline. This is the final structure passed to the LLM for
violation analysis.

Sources:
    - VideoAnalysisResult  → person count changes over time
    - ObstructionResult    → camera blocking events
    - DiarizationResult    → who said what and when
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.video.frame_analysis import VideoAnalysisResult
from src.video.camera_obstruction import ObstructionResult
from src.speech.diarization import DiarizationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    SPEECH          = "speech"           # someone spoke
    PERSON_CHANGE   = "person_change"    # number of people in frame changed
    OBSTRUCTION     = "obstruction"      # camera was blocked


@dataclass
class TimelineEvent:
    """A single event on the timeline."""
    event_type: EventType
    start: float                        # seconds
    end: float                          # seconds
    description: str                    # human-readable, fed to LLM
    speaker: Optional[str] = None       # for SPEECH events
    text: Optional[str] = None          # spoken text, for SPEECH events
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        d = {
            "type": self.event_type.value,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": self.duration,
            "description": self.description,
        }
        if self.speaker:
            d["speaker"] = self.speaker
        if self.text:
            d["text"] = self.text
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Timeline:
    """
    Full chronological timeline of all events in the video.
    This is the main input to llm/analyzer.py.
    """
    file_path: str
    duration: float
    events: list[TimelineEvent] = field(default_factory=list)

    @property
    def speech_events(self) -> list[TimelineEvent]:
        return [e for e in self.events if e.event_type == EventType.SPEECH]

    @property
    def obstruction_events(self) -> list[TimelineEvent]:
        return [e for e in self.events if e.event_type == EventType.OBSTRUCTION]

    @property
    def person_change_events(self) -> list[TimelineEvent]:
        return [e for e in self.events if e.event_type == EventType.PERSON_CHANGE]

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "duration": round(self.duration, 2),
            "total_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def to_text(self) -> str:
        """
        Plain text representation for LLM prompt injection.
        Format: [MM:SS] TYPE: description
        """
        lines = [f"Видео: {self.file_path} (длительность: {self._fmt(self.duration)})\n"]
        for event in self.events:
            time_str = f"[{self._fmt(event.start)}–{self._fmt(event.end)}]"
            line = f"{time_str} {event.event_type.value.upper()}: {event.description}"
            if event.text:
                line += f'\n    "{event.text}"'
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _fmt(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TimelineBuilder:
    """
    Builds a Timeline from all analysis results.

    Usage:
        from src.video.timeline_events import TimelineBuilder

        timeline = TimelineBuilder().build(
            video_result=video_result,
            obstruction_result=obstruction_result,
            diarization_result=diarization_result,
        )
        print(timeline.to_text())  # → pass to LLM
    """

    # Minimum person count change to register as an event
    PERSON_CHANGE_THRESHOLD = 1

    def build(
        self,
        video_result: VideoAnalysisResult,
        obstruction_result: ObstructionResult,
        diarization_result: DiarizationResult,
    ) -> Timeline:
        """
        Aggregate all events into a single sorted timeline.

        Args:
            video_result:        From frame_analysis.py
            obstruction_result:  From camera_obstruction.py
            diarization_result:  From diarization.py

        Returns:
            Timeline sorted chronologically.
        """
        events: list[TimelineEvent] = []

        events.extend(self._build_person_events(video_result))
        events.extend(self._build_obstruction_events(obstruction_result))
        events.extend(self._build_speech_events(diarization_result))

        # Sort by start time, then by type priority (speech last — most detail)
        type_priority = {
            EventType.PERSON_CHANGE: 0,
            EventType.OBSTRUCTION: 1,
            EventType.SPEECH: 2,
        }
        events.sort(key=lambda e: (e.start, type_priority[e.event_type]))

        timeline = Timeline(
            file_path=video_result.file_path,
            duration=video_result.duration,
            events=events,
        )

        logger.info(
            "Timeline built: %d events (%d speech, %d obstructions, %d person changes)",
            len(events),
            len(timeline.speech_events),
            len(timeline.obstruction_events),
            len(timeline.person_change_events),
        )

        return timeline

    # ------------------------------------------------------------------
    # Event builders
    # ------------------------------------------------------------------

    def _build_person_events(
        self,
        video_result: VideoAnalysisResult,
    ) -> list[TimelineEvent]:
        """Detect moments when the number of people in frame changes."""
        events: list[TimelineEvent] = []
        frames = video_result.frames

        if not frames:
            return events

        prev_count = frames[0].person_count

        for frame in frames[1:]:
            delta = frame.person_count - prev_count
            if abs(delta) >= self.PERSON_CHANGE_THRESHOLD:
                direction = "вошёл(и)" if delta > 0 else "вышел(и)"
                events.append(TimelineEvent(
                    event_type=EventType.PERSON_CHANGE,
                    start=frame.timestamp,
                    end=frame.timestamp,
                    description=(
                        f"Количество людей в кадре изменилось: "
                        f"{prev_count} → {frame.person_count} ({direction})"
                    ),
                    metadata={
                        "prev_count": prev_count,
                        "new_count": frame.person_count,
                        "delta": delta,
                    },
                ))
                prev_count = frame.person_count

        return events

    @staticmethod
    def _build_obstruction_events(
        obstruction_result: ObstructionResult,
    ) -> list[TimelineEvent]:
        """Convert ObstructionEvent list to TimelineEvent list."""
        reason_labels = {
            "coverage": "перекрытие объектива",
            "darkness":  "затемнение камеры",
            "blur":      "размытие объектива",
        }
        events = []
        for obs in obstruction_result.events:
            label = reason_labels.get(obs.reason, obs.reason)
            events.append(TimelineEvent(
                event_type=EventType.OBSTRUCTION,
                start=obs.start,
                end=obs.end,
                description=(
                    f"Камера заблокирована ({label}), "
                    f"длительность {obs.duration}с, "
                    f"уверенность {obs.confidence:.0%}"
                ),
                metadata={
                    "reason": obs.reason,
                    "confidence": obs.confidence,
                },
            ))
        return events

    @staticmethod
    def _build_speech_events(
        diarization_result: DiarizationResult,
    ) -> list[TimelineEvent]:
        """Convert Utterance list to TimelineEvent list."""
        events = []
        for utt in diarization_result.utterances:
            events.append(TimelineEvent(
                event_type=EventType.SPEECH,
                start=utt.start,
                end=utt.end,
                description=f"{utt.speaker} говорит",
                speaker=utt.speaker,
                text=utt.text,
                metadata={"confidence": utt.confidence},
            ))
        return events


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_timeline(
    video_result: VideoAnalysisResult,
    obstruction_result: ObstructionResult,
    diarization_result: DiarizationResult,
) -> Timeline:
    """
    One-liner for use in analysis_pipeline.py.

    Example:
        from src.video.timeline_events import build_timeline

        timeline = build_timeline(video_result, obstruction_result, diarization_result)

        # plain text for LLM
        print(timeline.to_text())

        # dict for JSON export
        data = timeline.to_dict()
    """
    return TimelineBuilder().build(
        video_result=video_result,
        obstruction_result=obstruction_result,
        diarization_result=diarization_result,
    )