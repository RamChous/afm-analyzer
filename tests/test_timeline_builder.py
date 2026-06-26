"""
tests/test_timeline_builder.py
Тесты агрегатора таймлайна — без загрузки моделей.
Запуск: pytest tests/test_timeline_builder.py -v
"""

from src.video.timeline_events import TimelineBuilder
from src.models.video import (
    Detection, FrameResult, VideoAnalysisResult,
    ObstructionEvent, ObstructionResult,
    EventType,
)
from src.models.speech import Utterance, DiarizationResult


# ---------------------------------------------------------------------------
# Фабрики фейковых данных
# ---------------------------------------------------------------------------

def person(n: int) -> list[Detection]:
    """n детекций-людей (координаты неважны для timeline)."""
    return [Detection(label="person", confidence=0.9,
                      x1=0.0, y1=0.0, x2=0.1, y2=0.1) for _ in range(n)]


def frame(idx: int, ts: float, n_persons: int) -> FrameResult:
    return FrameResult(frame_index=idx, timestamp=ts, detections=person(n_persons))


def video(frames: list[FrameResult], duration: float = 60.0) -> VideoAnalysisResult:
    return VideoAnalysisResult(
        file_path="fake.mp4", fps=1.0,
        total_frames=int(duration), duration=duration, frames=frames,
    )


def utt(speaker: str, start: float, end: float, text: str = "x") -> Utterance:
    return Utterance(speaker=speaker, start=start, end=end, text=text, confidence=0.9)


def diar(utterances: list[Utterance]) -> DiarizationResult:
    return DiarizationResult(
        file_path="fake.mp4",
        num_speakers=len({u.speaker for u in utterances}),
        utterances=utterances,
    )


# ---------------------------------------------------------------------------
# person_count_change: событие только на ИЗМЕНЕНИЕ
# ---------------------------------------------------------------------------

def test_person_count_events_only_on_change():
    """Одинаковое число людей подряд → НИ одного события, потом одно на изменение."""
    frames = [
        frame(0, 0.0, 2),
        frame(1, 1.0, 2),   # без изменений → нет события
        frame(2, 2.0, 2),   # без изменений → нет события
        frame(3, 3.0, 3),   # 2→3 → событие
        frame(4, 4.0, 1),   # 3→1 → событие
    ]
    timeline = TimelineBuilder().build(video(frames))
    pc = timeline.of_type(EventType.PERSON_COUNT_CHANGE)

    assert len(pc) == 2, "Ожидали ровно 2 изменения, а не событие на кадр"
    assert pc[0].timestamp == 3.0
    assert pc[0].data == {"from": 2, "to": 3}
    assert pc[1].timestamp == 4.0
    assert pc[1].data == {"from": 3, "to": 1}


def test_person_count_no_events_when_constant():
    """Число людей не меняется всё видео → нет person_count событий."""
    frames = [frame(i, float(i), 1) for i in range(5)]
    timeline = TimelineBuilder().build(video(frames))
    assert timeline.of_type(EventType.PERSON_COUNT_CHANGE) == []


# ---------------------------------------------------------------------------
# obstruction: интервальные события с end/duration
# ---------------------------------------------------------------------------

def test_obstruction_events_are_intervals():
    frames = [frame(0, 0.0, 1)]
    obstruction = ObstructionResult(
        file_path="fake.mp4",
        events=[
            ObstructionEvent(start=10.0, end=18.0, reason="coverage", confidence=0.8),
        ],
    )
    timeline = TimelineBuilder().build(video(frames), obstruction=obstruction)
    obs = timeline.of_type(EventType.OBSTRUCTION)

    assert len(obs) == 1
    assert obs[0].timestamp == 10.0
    assert obs[0].end == 18.0
    assert obs[0].duration == 8.0          # end - start
    assert obs[0].data["reason"] == "coverage"


# ---------------------------------------------------------------------------
# speaker_change: событие только при смене говорящего
# ---------------------------------------------------------------------------

def test_speaker_change_only_on_switch():
    """Подряд один спикер → одно событие; смена → новое событие."""
    utterances = [
        utt("SPEAKER_00", 0.0, 1.0),
        utt("SPEAKER_00", 1.2, 2.0),   # тот же → нет нового события
        utt("SPEAKER_01", 2.5, 3.0),   # смена → событие
        utt("SPEAKER_00", 3.5, 4.0),   # снова смена → событие
    ]
    timeline = TimelineBuilder().build(
        video([frame(0, 0.0, 2)]), diarization=diar(utterances),
    )
    sc = timeline.of_type(EventType.SPEAKER_CHANGE)

    assert len(sc) == 3                    # 00, 01, 00 — три смены подряд
    assert [e.data["speaker"] for e in sc] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert sc[0].timestamp == 0.0


# ---------------------------------------------------------------------------
# silence: паузы длиннее порога
# ---------------------------------------------------------------------------

def test_silence_events_above_threshold():
    """Пауза >= порога → событие SILENCE; короткая пауза → ничего."""
    utterances = [
        utt("SPEAKER_00", 0.0, 2.0),
        utt("SPEAKER_01", 4.0, 5.0),    # пауза 2с < 5 → нет события
        utt("SPEAKER_00", 15.0, 16.0),  # пауза 10с >= 5 → SILENCE
    ]
    timeline = TimelineBuilder(silence_threshold=5.0).build(
        video([frame(0, 0.0, 2)]), diarization=diar(utterances),
    )
    sil = timeline.of_type(EventType.SILENCE)

    assert len(sil) == 1
    assert sil[0].timestamp == 5.0         # конец предыдущей реплики
    assert sil[0].end == 15.0
    assert sil[0].duration == 10.0
    assert sil[0].data["gap_seconds"] == 10.0


# ---------------------------------------------------------------------------
# Сортировка и опциональные входы
# ---------------------------------------------------------------------------

def test_events_sorted_by_timestamp():
    """События из разных источников должны идти строго по времени."""
    frames = [frame(0, 0.0, 1), frame(1, 8.0, 2)]   # изменение на 8.0
    obstruction = ObstructionResult(
        file_path="fake.mp4",
        events=[ObstructionEvent(start=3.0, end=5.0, reason="darkness", confidence=0.9)],
    )
    utterances = [utt("SPEAKER_00", 1.0, 2.0), utt("SPEAKER_01", 6.0, 7.0)]

    timeline = TimelineBuilder().build(
        video(frames, duration=20.0),
        obstruction=obstruction,
        diarization=diar(utterances),
    )
    times = [e.timestamp for e in timeline.events]
    assert times == sorted(times), f"События не отсортированы: {times}"


def test_works_without_optional_inputs():
    """Только видео, без obstruction и diarization — не должно падать."""
    frames = [frame(0, 0.0, 1), frame(1, 1.0, 2)]
    timeline = TimelineBuilder().build(video(frames))

    assert timeline.of_type(EventType.PERSON_COUNT_CHANGE)      # есть
    assert timeline.of_type(EventType.OBSTRUCTION) == []         # нет
    assert timeline.of_type(EventType.SPEAKER_CHANGE) == []      # нет
    assert timeline.of_type(EventType.SILENCE) == []             # нет


def test_empty_video_produces_empty_timeline():
    """Видео без кадров → пустой таймлайн, без исключений."""
    timeline = TimelineBuilder().build(video([], duration=0.0))
    assert timeline.events == []
    assert timeline.to_dict()["event_count"] == 0