import os
import math
import logging
import tempfile
from pathlib import Path
from typing import Optional

import ffmpeg
import torch
from faster_whisper import WhisperModel

from src.models.speech import (
    TranscriptionWord,
    TranscriptionSegment,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class Transcriber:

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

    def __init__(
        self,
        model_size: str = "large-v3",
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 10,
        word_timestamps: bool = True,
        vad_filter: bool = False,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self._model = None

        logger.info(
            "Transcriber initialized (model=%s, language=%s, device=%s, compute=%s)",
            model_size, language or "auto", self.device, self.compute_type,
        )

    # ========================================================
    # Public API
    # ========================================================

    def transcribe(self, file_path: str) -> TranscriptionResult:
        path = Path(file_path)
        self._validate_file(path)
        logger.info("Starting transcription: %s", path.name)

        if path.suffix.lower() in self.VIDEO_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name
            try:
                self._extract_audio(str(path), audio_path)
                result = self._run_whisper(audio_path, source_path=str(path))
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        else:
            result = self._run_whisper(str(path), source_path=str(path))

        logger.info(
            "Transcription complete: %d segments, language=%s, duration=%.1fs",
            len(result.segments), result.language, result.duration or 0.0,
        )
        return result

    def transcribe_batch(self, file_paths: list[str]) -> list[TranscriptionResult]:
        return [self.transcribe(fp) for fp in file_paths]

    # ========================================================
    # Internal
    # ========================================================

    def _load_model(self):
        if self._model is None:
            logger.info("Loading Faster-Whisper model: %s", self.model_size)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Model loaded.")
        return self._model

    def _validate_file(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {path.suffix}")

    @staticmethod
    def _extract_audio(video_path: str, output_wav: str):
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
            raise RuntimeError(f"FFmpeg failed:\n{stderr}") from exc

    def _run_whisper(self, audio_path: str, source_path: str) -> TranscriptionResult:
        model = self._load_model()

        segments_generator, info = model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=self.word_timestamps,
            condition_on_previous_text=False,
        )

        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []

        for segment in segments_generator:
            text = segment.text.strip()
            full_text_parts.append(text)

            words = [
                TranscriptionWord(
                    start=round(w.start, 2),
                    end=round(w.end, 2),
                    word=w.word,
                    probability=round(float(w.probability), 4),
                )
                for w in (segment.words or [])
            ] if self.word_timestamps else []

            segments.append(TranscriptionSegment(
                start=round(segment.start, 2),
                end=round(segment.end, 2),
                text=text,
                # avg_logprob (лог-вероятность) → честное 0..1 через exp()
                confidence=round(math.exp(segment.avg_logprob), 4),
                words=words,
            ))

        return TranscriptionResult(
            file_path=source_path,
            full_text=" ".join(full_text_parts),
            segments=segments,
            language=info.language,
            duration=round(info.duration, 2),   # точная длительность аудио
            model_used=self.model_size,
        )


# ============================================================
# Convenience function
# ============================================================

def transcribe_video(
    file_path: str,
    model_size: str = "large-v3",
    language: Optional[str] = None,
) -> TranscriptionResult:
    return Transcriber(model_size=model_size, language=language).transcribe(file_path)