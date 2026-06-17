import os
import logging
import tempfile

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import ffmpeg
import torch

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


# ============================================================
# Models
# ============================================================

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.0


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
                }
                for s in self.segments
            ],
        }


# ============================================================
# Transcriber
# ============================================================

class Transcriber:

    SUPPORTED_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".flac",
    }

    def __init__(
        self,
        model_size: str = "large-v3",
        language: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):

        self.model_size = model_size
        self.language = language
        self.device = device

        self._model = None

        logger.info(
            "Transcriber initialized "
            "(model=%s, language=%s, device=%s)",
            model_size,
            language or "auto",
            device,
        )

    # ========================================================
    # Public API
    # ========================================================

    def transcribe(self, file_path: str) -> TranscriptionResult:

        path = Path(file_path)

        self._validate_file(path)

        logger.info("Starting transcription: %s", path.name)

        video_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
        }

        if path.suffix.lower() in video_extensions:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as tmp:

                audio_path = tmp.name

            try:

                self._extract_audio(
                    str(path),
                    audio_path,
                )

                result = self._run_whisper(
                    audio_path,
                    source_path=str(path),
                )

            finally:

                if os.path.exists(audio_path):
                    os.remove(audio_path)

        else:

            result = self._run_whisper(
                str(path),
                source_path=str(path),
            )

        return result

    def transcribe_batch(
        self,
        file_paths: list[str],
    ) -> list[TranscriptionResult]:

        results = []

        for file_path in file_paths:

            results.append(
                self.transcribe(file_path)
            )

        return results

    # ========================================================
    # Internal
    # ========================================================

    def _load_model(self):

        if self._model is None:
            compute_type=(
                "float16"
                if self.device == "cuda"
                else "int8"
            )

            logger.info(
                "Loading Faster-Whisper model: %s",
                self.model_size,
            )

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )

            logger.info("Model loaded.")

        return self._model

    def _validate_file(self, path: Path):

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}"
            )

    @staticmethod
    def _extract_audio(
        video_path: str,
        output_wav: str,
    ):

        try:
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_wav,
                    ac=1,
                    ar=16000,
                    acodec="pcm_s16le"
                )
                .overwrite_output()
                .run(quiet=True)
            )

        except ffmpeg.Error as exc:

            stderr = (
                exc.stderr.decode()
                if exc.stderr
                else str(exc)
            )

            raise RuntimeError(
                f"FFmpeg failed:\n{stderr}"
            )

    def _run_whisper(
        self,
        audio_path: str,
        source_path: str,
    ) -> TranscriptionResult:

        model = self._load_model()

        segments_generator, info = model.transcribe(
            audio_path,
            language=self.language,
            vad_filter=False,
            beam_size=10,
            condition_on_previous_text=False,
        )

        segments = []

        full_text_parts = []

        for segment in segments_generator:

            text = segment.text.strip()

            full_text_parts.append(text)

            segments.append(
                TranscriptionSegment(
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=text,
                    confidence=float(segment.avg_logprob),
                )
            )

        full_text = " ".join(full_text_parts)

        return TranscriptionResult(
            file_path=source_path,
            full_text=full_text,
            segments=segments,
            language=info.language,
            duration=segments[-1].end if segments else 0.0,
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

    transcriber = Transcriber(
        model_size=model_size,
        language=language,
    )

    return transcriber.transcribe(file_path)          