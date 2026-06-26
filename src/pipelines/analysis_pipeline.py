"""
src/pipelines/analysis_pipeline.py
Сквозной пайплайн анализа видеозаписи обыска.

Стадии (каждая дорогая — кэшируется на диск):
    1. frame_analysis   — YOLO по кадрам (video)
    2. person_detection — счётчик людей → подсказка для диаризации
    3. obstruction      — перекрытие камеры
    4. extract_audio    — аудио из видео ОДИН раз (переиспользуется ниже)
    5. transcription    — Whisper
    6. diarization      — pyannote (использует тот же wav + подсказку по спикерам)
    7. timeline         — сборка единой шкалы событий (без модели)
    8. llm_analysis     — поиск нарушений (Ollama)
    9. reports          — JSON + Excel

Кэш: результат каждой дорогой стадии пиклится в cache_dir. При повторном
запуске стадия, чей кэш валиден, не пересчитывается. Падение поздней стадии
не теряет результаты ранних.
"""

import os
import pickle
import hashlib
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any

from src.speech.transcription import Transcriber
from src.speech.diarization import Diarizer
from src.video.frame_analysis import FrameAnalyzer
from src.video.person_detection import PersonDetector
from src.video.camera_obstruction import CameraObstructionDetector
from src.video.timeline_events import TimelineBuilder
from src.llm.client import LLMClient
from src.llm.rules import parse_rules
from src.llm.analyzer import Analyzer
from src.reports.json_export import export_json
from src.reports.excel_export import export_excel
from src.models.llm import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Все настройки прогона в одном месте."""
    # LLM
    llm_model: str = "qwen3:14b"
    llm_num_ctx: int = 8192
    hf_token: str = ""                      # из .env, не хардкод

    # пути
    rules_path: str = "config/procedures/search_procedures.txt"
    output_dir: str = "output"
    cache_dir: str = ".cache"

    # speech
    whisper_model: str = "large-v3"
    language: Optional[str] = "ru"          # домен русскоязычный → не auto

    # video
    yolo_model: str = "n"
    sample_rate: float = 3.0

    # поведение
    use_cache: bool = True
    use_speaker_hint: bool = False          # CV-подсказка спикеров — по умолчанию off
    min_speakers: Optional[int] = 2         # допрос: минимум следователь+обыскиваемый


@dataclass
class PipelineResult:
    """Итог прогона: что получилось и где лежит."""
    file_path: str
    analysis: Optional[AnalysisResult] = None
    json_path: str = ""
    excel_path: str = ""
    stages_completed: list[str] = field(default_factory=list)
    stages_failed: list[str] = field(default_factory=list)


class AnalysisPipeline:
    """
    Usage:
        from dotenv import load_dotenv
        load_dotenv()

        cfg = PipelineConfig(
            llm_model="qwen3:14b",
            hf_token=os.environ["HF_TOKEN"],
        )
        pipeline = AnalysisPipeline(cfg)
        result = pipeline.run("video.mp4")
        print(result.json_path, result.excel_path)
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, file_path: str) -> PipelineResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        result = PipelineResult(file_path=str(path))
        logger.info("=== Пайплайн запущен: %s ===", path.name)

        # --- ВИДЕО ----------------------------------------------------
        video_result = self._stage(
            result, "frame_analysis", file_path,
            lambda: FrameAnalyzer(
                model_size=self.cfg.yolo_model,
                sample_rate=self.cfg.sample_rate,
                classes=[0],
            ).process(file_path),
        )

        persons = None
        obstruction = None
        if video_result is not None:
            persons = self._stage(
                result, "person_detection", file_path,
                lambda: PersonDetector().count(video_result),
            )
            obstruction = self._stage(
                result, "obstruction", file_path,
                lambda: CameraObstructionDetector().detect(video_result),
            )

        # --- АУДИО (извлекаем ОДИН раз) -------------------------------
        audio_path, tmp_created = self._prepare_audio(path)

        transcription = None
        diarization = None
        try:
            transcription = self._stage(
                result, "transcription", file_path,
                lambda: Transcriber(
                    model_size=self.cfg.whisper_model,
                    language=self.cfg.language,
                ).transcribe(audio_path),
            )

            if transcription is not None:
                speaker_hint = None
                if self.cfg.use_speaker_hint and persons is not None:
                    speaker_hint = persons.max_count or None

                diarization = self._stage(
                    result, "diarization", file_path,
                    lambda: Diarizer(hf_token=self.cfg.hf_token).diarize(
                        file_path=file_path,
                        transcription=transcription,
                        speaker_count=speaker_hint,
                        audio_path=audio_path,      # ← переиспользуем wav
                    ),
                )
        finally:
            if tmp_created and os.path.exists(audio_path):
                os.remove(audio_path)

        # --- TIMELINE -------------------------------------------------
        timeline = None
        if video_result is not None:
            timeline = self._stage(
                result, "timeline", file_path,
                lambda: TimelineBuilder().build(
                    video_result=video_result,
                    obstruction=obstruction,
                    diarization=diarization,
                ),
            )

        # --- LLM-АНАЛИЗ -----------------------------------------------
        if diarization is None:
            logger.error("Нет диаризации — LLM-анализ невозможен. Останов.")
            return result

        analysis = self._stage(
            result, "llm_analysis", file_path,
            lambda: self._run_llm(file_path, diarization, timeline),
        )
        if analysis is None:
            return result
        result.analysis = analysis

        # --- ОТЧЁТЫ ---------------------------------------------------
        stem = path.stem
        try:
            result.json_path = export_json(
                analysis, str(Path(self.cfg.output_dir) / f"{stem}_report.json"),
            )
            result.excel_path = export_excel(
                analysis, str(Path(self.cfg.output_dir) / f"{stem}_report.xlsx"),
            )
            result.stages_completed.append("reports")
        except Exception as exc:
            logger.exception("Ошибка экспорта отчётов: %s", exc)
            result.stages_failed.append("reports")

        logger.info("=== Пайплайн завершён. Готово: %s. Сбои: %s ===",
                    result.stages_completed, result.stages_failed or "нет")
        return result

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _run_llm(self, file_path, diarization, timeline) -> AnalysisResult:
        client = LLMClient(model=self.cfg.llm_model, num_ctx=self.cfg.llm_num_ctx)
        if not client.health_check():
            raise RuntimeError(
                f"Ollama недоступна или модель '{self.cfg.llm_model}' не загружена. "
                f"Проверь: ollama list / ollama pull {self.cfg.llm_model}"
            )
        rules = parse_rules(self.cfg.rules_path)
        if not rules:
            raise RuntimeError(f"Не загружено ни одного правила из {self.cfg.rules_path}")
        return Analyzer(client, rules).analyze(
            file_path=file_path, diarization=diarization, timeline=timeline,
        )

    # ------------------------------------------------------------------
    # Аудио
    # ------------------------------------------------------------------

    def _prepare_audio(self, path: Path) -> tuple[str, bool]:
        """
        Возвращает (audio_path, был_ли_создан_временный_файл).
        Видео → извлекаем wav один раз. Аудио → используем как есть.
        """
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
        if path.suffix.lower() not in video_exts:
            return str(path), False

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav = tmp.name
        logger.info("Извлекаю аудио (один раз) → %s", wav)
        Transcriber._extract_audio(str(path), wav)
        return wav, True

    # ------------------------------------------------------------------
    # Кэш + обёртка стадии
    # ------------------------------------------------------------------

    def _stage(
        self,
        result: PipelineResult,
        name: str,
        file_path: str,
        compute: Callable[[], Any],
    ) -> Any:
        """
        Выполняет стадию с кэшем и изоляцией сбоев.
        - кэш есть и валиден → загружаем, не считаем;
        - стадия упала → лог + пометка в stages_failed, возвращаем None
          (поздние стадии сами решат, могут ли работать без неё).
        """
        cache_file = self._cache_path(name, file_path)

        if self.cfg.use_cache and cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.info("[%s] загружено из кэша", name)
                result.stages_completed.append(name)
                return data
            except Exception as exc:
                logger.warning("[%s] кэш повреждён (%s) — пересчитываю", name, exc)

        logger.info("[%s] выполняю...", name)
        try:
            data = compute()
        except Exception as exc:
            logger.exception("[%s] СБОЙ: %s", name, exc)
            result.stages_failed.append(name)
            return None

        if self.cfg.use_cache:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(data, f)
            except Exception as exc:
                logger.warning("[%s] не удалось записать кэш: %s", name, exc)

        result.stages_completed.append(name)
        return data

    def _cache_path(self, stage: str, file_path: str) -> Path:
        """
        Ключ кэша = stage + путь + размер + mtime файла + ключевые параметры.
        Если файл изменился или поменялись параметры — кэш инвалидируется.
        """
        p = Path(file_path)
        stat = p.stat()
        params = f"{self.cfg.whisper_model}|{self.cfg.language}|{self.cfg.yolo_model}|{self.cfg.sample_rate}"
        key = f"{stage}|{p.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{params}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return Path(self.cfg.cache_dir) / f"{stage}_{digest}.pkl"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_search_video(file_path: str, config: PipelineConfig) -> PipelineResult:
    return AnalysisPipeline(config).run(file_path)