"""
timeline_events.py
Builds a single, time-sorted timeline of events from the outputs of all
other analyzers (frame_analysis, camera_obstruction, diarization).

Does NOT run any models — pure aggregation. The resulting Timeline is the
chronological view fed to the LLM analyzer.
"""

import logging
from typing import Optional

from src.models.video import (
    VideoAnalysisResult,
    ObstructionResult,
    EventType,
    TimelineEvent,
    Timeline,
)
from src.models.speech import DiarizationResult

logger = logging.getLogger(__name__)


class TimelineBuilder:
    """
    Merges per-analyzer results into one chronological Timeline.

    Usage:
        builder = TimelineBuilder()
        timeline = builder.build(
            video_result=video_result,
            obstruction=obstruction_result,   # optional
            diarization=diarization_result,   # optional
        )
        for ev in timeline.events:
            print(ev.timestamp, ev.event_type, ev.description)
    """

    def __init__(self, silence_threshold: float = 5.0):
        """
        Args:
            silence_threshold: паузы между репликами длиннее этого (сек)
                               становятся событиями SILENCE.
        """
        self.silence_threshold = silence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        video_result: VideoAnalysisResult,
        obstruction: Optional[ObstructionResult] = None,
        diarization: Optional[DiarizationResult] = None,
    ) -> Timeline:
        events: list[TimelineEvent] = []

        events.extend(self._person_count_events(video_result))

        if obstruction is not None:
            events.extend(self._obstruction_events(obstruction))

        if diarization is not None:
            events.extend(self._speaker_change_events(diarization))
            events.extend(self._silence_events(diarization, video_result.duration))

        # Единая сортировка по времени
        events.sort(key=lambda e: e.timestamp)

        logger.info("Timeline built: %d events over %.1fs",
                    len(events), video_result.duration)

        return Timeline(
            file_path=video_result.file_path,
            duration=video_result.duration,
            events=events,
        )

    # ------------------------------------------------------------------
    # Event generators
    # ------------------------------------------------------------------

    def _person_count_events(
        self,
        video_result: VideoAnalysisResult,
    ) -> list[TimelineEvent]:
        """Событие на каждое ИЗМЕНЕНИЕ числа людей в кадре (не на каждый кадр)."""
        events: list[TimelineEvent] = []
        prev_count: Optional[int] = None

        for frame in video_result.frames:
            count = frame.person_count
            if prev_count is None:
                prev_count = count
                continue
            if count != prev_count:
                direction = "увеличилось" if count > prev_count else "уменьшилось"
                events.append(TimelineEvent(
                    timestamp=frame.timestamp,
                    event_type=EventType.PERSON_COUNT_CHANGE,
                    description=(
                        f"Число людей в кадре {direction}: "
                        f"{prev_count} → {count}"
                    ),
                    data={"from": prev_count, "to": count},
                ))
                prev_count = count

        return events

    def _obstruction_events(
        self,
        obstruction: ObstructionResult,
    ) -> list[TimelineEvent]:
        """Каждый интервал перекрытия камеры → интервальное событие."""
        events: list[TimelineEvent] = []
        for ev in obstruction.events:
            events.append(TimelineEvent(
                timestamp=ev.start,
                end=ev.end,
                event_type=EventType.OBSTRUCTION,
                description=(
                    f"Камера перекрыта ({ev.reason}) на {ev.duration:.1f}с"
                ),
                data={"reason": ev.reason, "confidence": round(ev.confidence, 3)},
            ))
        return events

    def _speaker_change_events(
        self,
        diarization: DiarizationResult,
    ) -> list[TimelineEvent]:
        """Событие, когда говорящий сменился относительно предыдущей реплики."""
        events: list[TimelineEvent] = []
        prev_speaker: Optional[str] = None

        for utt in diarization.utterances:
            if utt.speaker != prev_speaker:
                events.append(TimelineEvent(
                    timestamp=utt.start,
                    event_type=EventType.SPEAKER_CHANGE,
                    description=f"Говорит {utt.speaker}",
                    data={"speaker": utt.speaker},
                ))
                prev_speaker = utt.speaker
        return events

    def _silence_events(
        self,
        diarization: DiarizationResult,
        video_duration: float,
    ) -> list[TimelineEvent]:
        """Длинные паузы между репликами → события SILENCE."""
        events: list[TimelineEvent] = []
        utts = sorted(diarization.utterances, key=lambda u: u.start)

        for prev, nxt in zip(utts, utts[1:]):
            gap = nxt.start - prev.end
            if gap >= self.silence_threshold:
                events.append(TimelineEvent(
                    timestamp=prev.end,
                    end=nxt.start,
                    event_type=EventType.SILENCE,
                    description=f"Тишина {gap:.1f}с (нет речи)",
                    data={"gap_seconds": round(gap, 2)},
                ))
        return events


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_timeline(
    video_result: VideoAnalysisResult,
    obstruction: Optional[ObstructionResult] = None,
    diarization: Optional[DiarizationResult] = None,
    silence_threshold: float = 5.0,
) -> Timeline:
    """
    One-liner for use in analysis_pipeline.py.

    Example:
        from src.video.timeline_events import build_timeline

        timeline = build_timeline(
            video_result=video_result,
            obstruction=obstruction_result,
            diarization=diarization_result,
        )
    """
    return TimelineBuilder(silence_threshold=silence_threshold).build(
        video_result=video_result,
        obstruction=obstruction,
        diarization=diarization,
    )