"""
transcription.py
Converts video/audio files to text using OpenAI Whisper.
"""

import os
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

import whisper
import ffmpeg
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionSegment:
    """Single timed segment from Whisper output."""
    start: float          # seconds
    end: float            # seconds
    text: str
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Full transcription result for one file."""
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


# ---------------------------------------------------------------------------
# Core transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Extracts audio from a video file and transcribes it with Whisper.

    Usage:
        transcriber = Transcriber(model_size="medium")
        result = transcriber.transcribe("path/to/video.mp4")
        print(result.full_text)
    """

    SUPPORTED_EXTENSIONS = {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",  # video
        ".mp3", ".wav", ".m4a", ".ogg", ".flac",          # audio
    }

    def __init__(
        self,
        model_size: str = "openai/whisper-large-v3-turbo",
        language: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_size: Whisper model size — tiny | base | small | medium | large.
                        'medium' is a good balance of speed and accuracy for Russian.
            language:   Force a language code (e.g. 'ru', 'kk'). None = auto-detect.
            device:     'cpu' or 'cuda'.
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self._pipeline: Optional[Any] = None

        logger.info("Transcriber initialised (model=%s, lang=%s, device=%s)",
                    model_size, language or "auto", device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, file_path: str) -> TranscriptionResult:
        """
        Transcribe a video or audio file.

        Args:
            file_path: Path to the media file.

        Returns:
            TranscriptionResult with full text and timed segments.
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info("Starting transcription: %s", path.name)

        # Extract audio to a temp WAV only when input is a video
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name
            try:
                self._extract_audio(str(path), audio_path)
                result = self._run_whisper(audio_path, source_path=str(path))
            finally:
                os.unlink(audio_path)
        else:
            result = self._run_whisper(str(path), source_path=str(path))

        logger.info("Transcription complete: %d characters, language=%s",
                    len(result.full_text), result.language)
        return result

    def transcribe_batch(self, file_paths: list[str]) -> list[TranscriptionResult]:
        """Transcribe multiple files sequentially."""
        results = []
        for i, fp in enumerate(file_paths, 1):
            logger.info("Batch transcription [%d/%d]: %s", i, len(file_paths), fp)
            try:
                results.append(self.transcribe(fp))
            except Exception as exc:
                logger.error("Failed to transcribe %s: %s", fp, exc)
                raise
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Lazy-load the Whisper model (cached after first call)."""
        if self._pipeline is None:
            logger.info("Loading Hugging Face pipeline for '%s'...", self.model_size)
            
            # Определяем точность вычислений в зависимости от доступности GPU
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            device_id = 0 if self.device == "cuda" else -1

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_size,
                torch_dtype=torch_dtype,
                device=device_id
            )
            logger.info("Hugging Face pipeline loaded successfully.")
        return self._pipeline

    def _validate_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

    @staticmethod
    def _extract_audio(video_path: str, output_wav: str) -> None:
        """
        Use ffmpeg to extract mono 16 kHz WAV audio from a video file.
        16 kHz mono is what Whisper natively expects.
        """
        logger.debug("Extracting audio: %s → %s", video_path, output_wav)
        try:
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_wav,
                    ac=1,           # mono
                    ar=16000,       # 16 kHz
                    acodec="pcm_s16le",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"ffmpeg failed to extract audio from '{video_path}': {exc.stderr.decode()}"
            ) from exc

    def _run_whisper(self, audio_path: str, source_path: str) -> TranscriptionResult:
        """Run Hugging Face inference pipeline and adapt result into dataclasses."""
        asr_pipe = self._load_model()

        # Настраиваем генерацию под требования Hugging Face
        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language

        # Запускаем инференс с чанкованием (chunk_length_s=30 спасает от переполнения ОЗУ)
        raw_result = asr_pipe(
            audio_path,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs=generate_kwargs
        )

        # Пересобираем чанки Hugging Face в ваши TranscriptionSegment структуры
        segments = []
        raw_chunks = raw_result.get("chunks", [])

        for chunk in raw_chunks:
            # Предотвращаем NoneType падения, если кодек потерял таймкод
            start_ts = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0
            end_ts = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else start_ts + 1.0

            segments.append(
                TranscriptionSegment(
                    start=round(float(start_ts), 2),
                    end=round(float(end_ts), 2),
                    text=chunk["text"].strip(),
                    confidence=0.0 # Пайплайн HF не возвращает avg_logprob напрямую
                )
            )

        full_text = raw_result.get("text", "").strip()

        return TranscriptionResult(
            file_path=source_path,
            full_text=full_text,
            segments=segments,
            language=self.language or "kazakh",
            duration=segments[-1].end if segments else 0.0,
            model_used=self.model_size,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def transcribe_video(
    file_path: str,
    model_size: str = "openai/whisper-large-v3-turbo",
    language: Optional[str] = None,
) -> TranscriptionResult:
    """
    One-liner helper for simple use cases.

    Example:
        result = transcribe_video("recording.mp4", language="ru")
        print(result.full_text)
    """
    return Transcriber(model_size=model_size, language=language).transcribe(file_path)