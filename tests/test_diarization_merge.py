"""
tests/test_diarization_merge.py
Тесты логики слияния диаризации — без загрузки моделей.
Запуск: pytest tests/test_diarization_merge.py -v
"""

from src.speech.diarization import Diarizer
from src.models.speech import (
    SpeakerTurn, TranscriptionSegment, TranscriptionWord,
)


def make_diarizer() -> Diarizer:
    # Токен не используется, пока не вызывается пайплайн.
    return Diarizer(hf_token="fake")


def word(start, end, text, prob=0.9):
    return TranscriptionWord(start=start, end=end, word=text, probability=prob)


def test_word_merge_splits_speakers_inside_one_segment():
    """Один Whisper-сегмент со сменой спикера внутри → 2 реплики."""
    d = make_diarizer()

    # Спикер 0 говорит 0–2с, спикер 1 говорит 2–4с
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=2.0),
        SpeakerTurn(speaker="SPEAKER_01", start=2.0, end=4.0),
    ]
    # Один сегмент Whisper на весь диапазон, но со словами
    seg = TranscriptionSegment(
        start=0.0, end=4.0, text="открой сейф нет ключа",
        words=[
            word(0.1, 0.6, "открой"),
            word(0.7, 1.2, "сейф"),
            word(2.1, 2.6, "нет"),
            word(2.7, 3.2, "ключа"),
        ],
    )

    utts = d._merge(turns, [seg])

    assert len(utts) == 2, "Ожидали разделение на двух спикеров"
    assert utts[0].speaker == "SPEAKER_00"
    assert utts[0].text == "открой сейф"
    assert utts[1].speaker == "SPEAKER_01"
    assert utts[1].text == "нет ключа"


def test_segment_fallback_when_no_words():
    """Без слов → посегментный режим, весь сегмент одному спикеру."""
    d = make_diarizer()
    turns = [SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=5.0)]
    seg = TranscriptionSegment(start=1.0, end=3.0, text="всё помещение", words=[])

    utts = d._merge(turns, [seg])

    assert len(utts) == 1
    assert utts[0].speaker == "SPEAKER_00"
    assert utts[0].text == "всё помещение"


def test_dominant_speaker_uses_summed_overlap():
    """Спикер с бОльшим СУММАРНЫМ перекрытием выигрывает, а не с одним длинным турном."""
    d = make_diarizer()
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.0),   # 1.0с
        SpeakerTurn(speaker="SPEAKER_01", start=1.0, end=2.5),   # один турн 1.5с
        SpeakerTurn(speaker="SPEAKER_00", start=2.5, end=4.0),   # ещё 1.5с → сумма 2.5с
    ]
    speaker, conf = d._dominant_speaker(0.0, 4.0, turns)

    assert speaker == "SPEAKER_00"   # 2.5с суммарно против 1.5с
    assert 0.0 < conf <= 1.0


def test_unknown_when_no_overlap():
    """Слово вне всех турнов и вне окна → UNKNOWN."""
    d = make_diarizer()
    turns = [SpeakerTurn(speaker="SPEAKER_00", start=10.0, end=20.0)]
    seg = TranscriptionSegment(
        start=0.0, end=1.0, text="привет",
        words=[word(0.1, 0.9, "привет")],
    )
    utts = d._merge(turns, [seg])

    assert len(utts) == 1
    assert utts[0].speaker == "UNKNOWN"


def test_gap_splits_same_speaker():
    """Один спикер, но большой разрыв между словами → две реплики."""
    d = make_diarizer()
    turns = [SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=30.0)]
    seg = TranscriptionSegment(
        start=0.0, end=25.0, text="начали ... продолжаем",
        words=[
            word(0.1, 0.6, "начали"),
            word(20.0, 20.6, "продолжаем"),   # разрыв ~19с > MAX_MERGE_GAP
        ],
    )
    utts = d._merge(turns, [seg])

    assert len(utts) == 2
    assert all(u.speaker == "SPEAKER_00" for u in utts)