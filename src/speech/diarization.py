"""
diarization.py
Splits audio by speakers (who spoke when) and merges with Whisper
transcription. Prefers word-level timestamps for precise speaker boundaries,
falls back to segment-level when words are unavailable.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import ffmpeg
import torch
from pyannote.audio import Pipeline
import soundfile as sf

from src.models.speech import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    SpeakerTurn,
    Utterance,
    DiarizationResult,
)

logger = logging.getLogger(__name__)

class Diarizer:
    """
    Speaker diarization using pyannote.audio.
    speaker_count from CV is treated as an UPPER BOUND (max_speakers) by
    default, because "max people in frame" is not the same as the number of
    speakers — someone may be silent on camera, or speaking off camera.

    Usage:
        diarizer = Diarizer(hf_token="hf_...")
        result = diarizer.diarize(
            file_path="video.mp4",
            transcription=transcription_result,
            speaker_count=3,            # hint from CV (used as max)
            audio_path="cached.wav",    # optional, skips re-extraction
        )
    """

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    # Below this overlap a word/segment is considered unassigned.
    MIN_OVERLAP_SECONDS = 0.05
    # Gap larger than this splits an utterance even for the same speaker.
    MAX_MERGE_GAP = 1.0

    def __init__(self, hf_token: str, device: Optional[str] = None):
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline: Optional[Pipeline] = None
        logger.info("Diarizer initialised (device=%s)", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(
        self,
        file_path: str,
        transcription: TranscriptionResult,
        speaker_count: Optional[int] = None,
        audio_path: Optional[str] = None,
    ) -> DiarizationResult:
        """
        Args:
            file_path:     Original video/audio path (for the result record).
            transcription: Result from Transcriber.transcribe().
            speaker_count: Hint from CV — used as max_speakers.
            audio_path:    Pre-extracted wav. If given, no re-extraction.
        """
        path = Path(file_path)
        logger.info("Starting diarization: %s (speaker_count<=%s)",
                    path.name, speaker_count or "auto")

        if audio_path is not None:
            turns = self._run_diarization(audio_path, speaker_count)
        elif path.suffix.lower() in self.VIDEO_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio = tmp.name
            try:
                self._extract_audio(str(path), tmp_audio)
                turns = self._run_diarization(tmp_audio, speaker_count)
            finally:
                if os.path.exists(tmp_audio):
                    os.unlink(tmp_audio)
        else:
            turns = self._run_diarization(str(path), speaker_count)

        utterances = self._merge(turns, transcription.segments)
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
        if self._pipeline is None:
            logger.info("Loading pyannote diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=self.hf_token,
            )
            self._pipeline.to(torch.device(self.device))
            logger.info("Pyannote pipeline loaded on %s.", self.device)
        return self._pipeline

    @staticmethod
    def _extract_audio(video_path: str, output_wav: str) -> None:
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_wav, ac=1, ar=16000, acodec="pcm_s16le")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode() if exc.stderr else str(exc)
            raise RuntimeError(f"ffmpeg failed on '{video_path}':\n{stderr}") from exc

    def _run_diarization(self, audio_path, speaker_count):
        pipeline = self._load_pipeline()
        kwargs = {}
        if speaker_count is not None:
            kwargs["max_speakers"] = speaker_count

        # Читаем wav через soundfile — без torchcodec/torchaudio
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile даёт (time, channels); pyannote ждёт (channels, time)
        waveform = torch.from_numpy(data.T)

        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **kwargs,
        )
        annotation = getattr(diarization, "speaker_diarization", diarization)

        turns = [
            SpeakerTurn(speaker=str(label), start=turn.start, end=turn.end)
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
        logger.debug("Raw speaker turns: %d", len(turns))
        return turns

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge(
        self,
        turns: list[SpeakerTurn],
        segments: list[TranscriptionSegment],
    ) -> list[Utterance]:
        """Word-level merge when words exist, else segment-level fallback."""
        has_words = any(seg.words for seg in segments)
        if has_words:
            return self._merge_by_words(turns, segments)
        logger.info("No word timestamps — falling back to segment-level merge.")
        return self._merge_by_segments(turns, segments)

    def _merge_by_words(
        self,
        turns: list[SpeakerTurn],
        segments: list[TranscriptionSegment],
    ) -> list[Utterance]:
        """
        Assign every word to the speaker active at its midpoint, then group
        consecutive same-speaker words into utterances. This cuts speaker
        boundaries inside long Whisper segments.
        """
        # Flatten all words across segments in time order.
        words: list[TranscriptionWord] = [
            w for seg in segments for w in seg.words
        ]
        if not words:
            return []
        words.sort(key=lambda w: w.start)

        utterances: list[Utterance] = []
        for w in words:
            speaker = self._speaker_at(0.5 * (w.start + w.end), turns) or "UNKNOWN"

            if (
                utterances
                and utterances[-1].speaker == speaker
                and (w.start - utterances[-1].end) < self.MAX_MERGE_GAP
            ):
                last = utterances[-1]
                last.text = (last.text + " " + w.word.strip()).strip()
                last.end = w.end
                last._probs.append(w.probability)  # type: ignore[attr-defined]
            else:
                utt = Utterance(
                    speaker=speaker,
                    start=w.start,
                    end=w.end,
                    text=w.word.strip(),
                    confidence=w.probability,
                )
                utt._probs = [w.probability]  # type: ignore[attr-defined]
                utterances.append(utt)

        # Finalise confidence as mean word probability.
        for utt in utterances:
            probs = getattr(utt, "_probs", [utt.confidence])
            utt.confidence = round(sum(probs) / len(probs), 4)
            if hasattr(utt, "_probs"):
                del utt._probs

        return utterances

    def _merge_by_segments(
        self,
        turns: list[SpeakerTurn],
        segments: list[TranscriptionSegment],
    ) -> list[Utterance]:
        """Fallback: assign each whole segment to its dominant speaker."""
        utterances: list[Utterance] = []
        for seg in segments:
            speaker, conf = self._dominant_speaker(seg.start, seg.end, turns)
            speaker = speaker or "UNKNOWN"

            if (
                utterances
                and utterances[-1].speaker == speaker
                and (seg.start - utterances[-1].end) < self.MAX_MERGE_GAP

            ):
                last = utterances[-1]
                last.text = (last.text + " " + seg.text).strip()
                last.end = seg.end
                last.confidence = round((last.confidence + conf) / 2, 4)
            else:
                utterances.append(Utterance(
                    speaker=speaker, start=seg.start, end=seg.end,
                    text=seg.text, confidence=round(conf, 4),
                ))
        return utterances

    def _speaker_at(self, t: float, turns: list[SpeakerTurn]) -> Optional[str]:
        """Speaker whose turn contains time t (closest turn if none contain it)."""
        for turn in turns:
            if turn.start <= t <= turn.end:
                return turn.speaker
        # Nothing contains t — pick the nearest turn within a small window.
        best, best_dist = None, self.MAX_MERGE_GAP
        for turn in turns:
            dist = turn.start - t if t < turn.start else t - turn.end
            if 0 <= dist < best_dist:
                best, best_dist = turn.speaker, dist
        return best

    def _dominant_speaker(
        self,
        start: float,
        end: float,
        turns: list[SpeakerTurn],
    ) -> tuple[Optional[str], float]:
        """Speaker with the largest SUMMED overlap over [start, end]."""
        overlaps: dict[str, float] = {}
        for turn in turns:
            ov = min(end, turn.end) - max(start, turn.start)
            if ov > 0:
                overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + ov

        if not overlaps:
            return None, 0.0

        speaker = max(overlaps, key=overlaps.get)
        total = overlaps[speaker]
        if total < self.MIN_OVERLAP_SECONDS:
            return None, 0.0

        duration = end - start
        return speaker, (total / duration if duration > 0 else 0.0)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def diarize_video(
    file_path: str,
    transcription: TranscriptionResult,
    hf_token: str,
    speaker_count: Optional[int] = None,
    audio_path: Optional[str] = None,
) -> DiarizationResult:

    return Diarizer(hf_token=hf_token).diarize(
        file_path=file_path,
        transcription=transcription,
        speaker_count=speaker_count,
        audio_path=audio_path,
    )