"""Data models for the speech layer (transcription + diarization)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptionWord:
    """Single word with its own timestamp (for precise diarization merge)."""
    start: float
    end: float
    word: str
    probability: float = 0.0


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.0          # 0..1, derived from avg_logprob via exp()
    words: list[TranscriptionWord] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    file_path: str
    full_text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "duration": self.duration,
            "model_used": self.model_used,
            "full_text": self.full_text,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "confidence": s.confidence,
                    "words": [
                        {"start": w.start, "end": w.end,
                         "word": w.word, "probability": w.probability}
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
        }


@dataclass
class SpeakerTurn:
    """Raw diarization turn before merging with transcription."""
    speaker: str
    start: float
    end: float


@dataclass
class Utterance:
    """A single speaker turn merged with transcription text."""
    speaker: str
    start: float
    end: float
    text: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class DiarizationResult:
    file_path: str
    num_speakers: int
    utterances: list[Utterance] = field(default_factory=list)
    speaker_turns: list[SpeakerTurn] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        return sorted({u.speaker for u in self.utterances})

    def utterances_by_speaker(self, speaker: str) -> list[Utterance]:
        return [u for u in self.utterances if u.speaker == speaker]

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "num_speakers": self.num_speakers,
            "speakers": self.speakers,
            "utterances": [u.to_dict() for u in self.utterances],
        }