"""
diarization.py
Splits audio by speakers (who spoke when) and merges the result
with Whisper transcription segments (who said what).

Flow:
    1. Receive speaker_count hint from CV (optional but improves accuracy)
    2. Run pyannote/speaker-diarization to get speaker turns
    3. Merge turns with TranscriptionResult segments from transcription.py
    4. Return DiarizationResult — list of utterances with speaker + text + timestamps
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ffmpeg
from pyannote.audio import Pipeline

from src.speech.transcription import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    """
    A single speaker turn merged with transcription text.
    This is the main unit consumed by violation checks.
    """
    speaker: str            # e.g. "SPEAKER_00", "SPEAKER_01"
    start: float            # seconds
    end: float              # seconds
    text: str               # transcribed text for this turn
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
class SpeakerTurn:
    """Raw diarization turn before merging with transcription."""
    speaker: str
    start: float
    end: float


@dataclass
class DiarizationResult:
    """Full result: speaker turns + merged utterances."""
    file_path: str
    num_speakers: int
    utterances: list[Utterance] = field(default_factory=list)
    speaker_turns: list[SpeakerTurn] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        """Unique speaker labels found in the file."""
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


# ---------------------------------------------------------------------------
# Core diarizer
# ---------------------------------------------------------------------------

class Diarizer:
    """
    Speaker diarization using pyannote.audio.

    Accepts an optional speaker_count hint — pass the value from CV
    (frame_analysis.py) to get more accurate speaker separation.

    Usage:
        diarizer = Diarizer(hf_token="hf_...")
        result = diarizer.diarize(
            file_path="video.mp4",
            transcription=transcription_result,
            speaker_count=3,   # from CV; omit to auto-detect
        )
        for utt in result.utterances:
            print(f"[{utt.speaker}] {utt.text}")
    """

    # Overlap threshold: a transcription segment is assigned to the speaker
    # who covers the most of its duration.
    MIN_OVERLAP_SECONDS = 0.1

    def __init__(self, hf_token: str):
        """
        Args:
            hf_token: HuggingFace token — required to download pyannote models.
                      Accept conditions at https://hf.co/pyannote/speaker-diarization-3.1
        """
        self.hf_token = hf_token
        self._pipeline: Optional[Pipeline] = None
        logger.info("Diarizer initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(
        self,
        file_path: str,
        transcription: TranscriptionResult,
        speaker_count: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Run diarization and merge with transcription.

        Args:
            file_path:     Path to the original video/audio file.
            transcription: Result from Transcriber.transcribe().
            speaker_count: Number of speakers (hint from CV).
                           If None — pyannote auto-detects.

        Returns:
            DiarizationResult with merged utterances.
        """
        path = Path(file_path)
        logger.info("Starting diarization: %s (speaker_count=%s)",
                    path.name, speaker_count or "auto")

        # Extract audio if video
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name
            try:
                self._extract_audio(str(path), audio_path)
                turns = self._run_diarization(audio_path, speaker_count)
            finally:
                os.unlink(audio_path)
        else:
            turns = self._run_diarization(str(path), speaker_count)

        utterances = self._merge_with_transcription(turns, transcription.segments)
        num_speakers = len({t.speaker for t in turns})

        logger.info("Diarization complete: %d speakers, %d utterances",
                    num_speakers, len(utterances))

        return DiarizationResult(
            file_path=str(path),
            num_speakers=num_speakers,
            utterances=utterances,
            speaker_turns=turns,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> Pipeline:
        """Lazy-load pyannote pipeline (cached after first call)."""
        if self._pipeline is None:
            logger.info("Loading pyannote diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
            )
            logger.info("Pyannote pipeline loaded.")
        return self._pipeline

    @staticmethod
    def _extract_audio(video_path: str, output_wav: str) -> None:
        """Extract mono 16 kHz WAV — same spec as transcription.py."""
        logger.debug("Extracting audio for diarization: %s", video_path)
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_wav, ac=1, ar=16000, acodec="pcm_s16le")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"ffmpeg failed on '{video_path}': {exc.stderr.decode()}"
            ) from exc

    def _run_diarization(
        self,
        audio_path: str,
        speaker_count: Optional[int],
    ) -> list[SpeakerTurn]:
        """Run pyannote and return raw speaker turns."""
        pipeline = self._load_pipeline()

        kwargs = {}
        if speaker_count is not None:
            kwargs["num_speakers"] = speaker_count

        diarization = pipeline(audio_path, **kwargs)

        # Вытаскиваем оригинальную аннотацию из нового DiarizeOutput
        if hasattr(diarization, "speaker_diarization"):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization

        # Теперь спокойно вызываем itertracks, так как annotation — это стандартный pyannote Annotation
        turns = [
            SpeakerTurn(
                speaker=str(label),
                start=turn.start,
                end=turn.end,
            )
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]

        logger.debug("Raw speaker turns: %d", len(turns))
        return turns

    def _merge_with_transcription(
        self,
        turns: list[SpeakerTurn],
        segments: list[TranscriptionSegment],
    ) -> list[Utterance]:
        """
        Assign each transcription segment to the speaker with the most
        temporal overlap, then group consecutive same-speaker segments
        into utterances.

        Strategy:
            For each Whisper segment, find all speaker turns that overlap
            with [segment.start, segment.end]. Assign the segment to the
            speaker whose overlap duration is the longest.
        """
        utterances: list[Utterance] = []

        for seg in segments:
            speaker, confidence = self._dominant_speaker(seg, turns)
            if speaker is None:
                speaker = "UNKNOWN"

            # Merge with previous utterance if same speaker and close in time
            if (
                utterances
                and utterances[-1].speaker == speaker
                and (seg.start - utterances[-1].end) < 1.0  # gap < 1 sec
            ):
                last = utterances[-1]
                last.text = (last.text + " " + seg.text).strip()
                last.end = seg.end
                last.confidence = (last.confidence + confidence) / 2
            else:
                utterances.append(Utterance(
                    speaker=speaker,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    confidence=confidence,
                ))

        return utterances

    def _dominant_speaker(
        self,
        seg: TranscriptionSegment,
        turns: list[SpeakerTurn],
    ) -> tuple[Optional[str], float]:
        """
        Find the speaker who covers the most of the segment's duration.
        Returns (speaker_label, overlap_fraction).
        """
        best_speaker: Optional[str] = None
        best_overlap = self.MIN_OVERLAP_SECONDS

        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        seg_duration = seg.end - seg.start
        confidence = (best_overlap / seg_duration) if seg_duration > 0 else 0.0

        return best_speaker, confidence


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def diarize_video(
    file_path: str,
    transcription: TranscriptionResult,
    hf_token: str,
    speaker_count: Optional[int] = None,
) -> DiarizationResult:
    """
    One-liner helper for use in analysis_pipeline.py.

    Example:
        from src.speech.transcription import transcribe_video
        from src.speech.diarization import diarize_video

        transcription = transcribe_video("video.mp4", language=None)
        diarization = diarize_video(
            file_path="video.mp4",
            transcription=transcription,
            hf_token="hf_...",
            speaker_count=3,   # pass value from frame_analysis.py
        )

        for utt in diarization.utterances:
            print(f"[{utt.speaker} {utt.start:.1f}s] {utt.text}")
    """
    return Diarizer(hf_token=hf_token).diarize(
        file_path=file_path,
        transcription=transcription,
        speaker_count=speaker_count,
    )