"""
app.py
Точка входа: анализ видеозаписи обыска от файла до отчётов.

Запуск:
    python app.py video.mp4
    python app.py video.mp4 --model qwen3:14b --language ru --no-cache
    python app.py *.mp4 --output-dir reports/

Перед запуском:
    1. Заполни .env:  HF_TOKEN=hf_...
    2. Подними Ollama: ollama serve  (и  ollama pull qwen3:30b)
    3. Системный ffmpeg в PATH:  ffmpeg -version
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.pipelines.analysis_pipeline import (
    AnalysisPipeline,
    PipelineConfig,
    PipelineResult,
)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # pyannote/torch шумят на INFO — приглушаем, если не verbose
    if not verbose:
        for noisy in ("pyannote", "torch", "speechbrain", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def build_config(args, hf_token: str) -> PipelineConfig:
    return PipelineConfig(
        llm_model=args.model,
        llm_num_ctx=args.num_ctx,
        hf_token=hf_token,
        rules_path=args.rules,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        whisper_model=args.whisper_model,
        language=args.language,
        yolo_model=args.yolo,
        sample_rate=args.sample_rate,
        use_cache=not args.no_cache,
        use_speaker_hint=args.speaker_hint,
        min_speakers=args.min_speakers,
    )


def print_result(result: PipelineResult) -> None:
    """Человекочитаемый итог одного файла."""
    name = Path(result.file_path).name
    print("\n" + "=" * 70)
    print(f"ФАЙЛ: {name}")
    print("=" * 70)

    if result.stages_failed:
        print(f"  ⚠ Сбойные стадии: {', '.join(result.stages_failed)}")

    if result.analysis is None:
        print("  ✗ Анализ не выполнен — отчёты не созданы.")
        return

    a = result.analysis
    print(f"  {a.summary}")
    if a.speaker_roles:
        roles = ", ".join(f"{k}={v}" for k, v in a.speaker_roles.items())
        print(f"  Роли: {roles}")

    # Топ нарушений по критичности
    if a.violations:
        print("  Нарушения:")
        for v in a.violations_by_severity():
            m, s = divmod(int(v.timestamp), 60)
            print(f"    [{m:02d}:{s:02d}] {v.severity.value.upper():<8} "
                  f"{v.rule_id}  (conf {v.confidence:.0%})")

    print(f"\n  JSON:  {result.json_path}")
    print(f"  Excel: {result.excel_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Анализ видеозаписи обыска: транскрипция, диаризация, "
                    "CV, поиск процессуальных нарушений.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help="Видео/аудио файл(ы) для анализа")

    # LLM
    parser.add_argument("--model", default="qwen3:14b",
                        help="Модель Ollama для анализа")
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="Контекстное окно LLM")

    # пути
    parser.add_argument("--rules", default="config/procedures/search_procedures.txt",
                        help="Файл правил")
    parser.add_argument("--output-dir", default="output", help="Папка для отчётов")
    parser.add_argument("--cache-dir", default=".cache", help="Папка кэша стадий")

    # speech
    parser.add_argument("--whisper-model", default="large-v3", help="Модель Whisper")
    parser.add_argument("--language", default="ru",
                        help="Код языка ('ru', 'kk') или 'auto' для автоопределения")

    # video
    parser.add_argument("--yolo", default="n", choices=["n", "s", "m", "l", "x"],
                        help="Размер модели YOLOv8")
    parser.add_argument("--sample-rate", type=float, default=3.0,
                        help="Кадров в секунду для анализа видео")

    # диаризация
    parser.add_argument("--min-speakers", type=int, default=2,
                        help="Минимум говорящих (для допроса осмысленно 2)")
    parser.add_argument("--speaker-hint", action="store_true",
                        help="Использовать число людей из CV как подсказку (по умолч. off)")

    # поведение
    parser.add_argument("--no-cache", action="store_true",
                        help="Не использовать кэш стадий (всё пересчитать)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Подробные логи (DEBUG)")

    args = parser.parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger("app")

    # .env из корня проекта (рядом с этим файлом)
    load_dotenv(Path(__file__).resolve().parent / ".env")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.error("Не задан HF_TOKEN. Создай файл .env с строкой:  HF_TOKEN=hf_...")
        return 1

    # 'auto' → None для Whisper
    if args.language.lower() == "auto":
        args.language = None

    config = build_config(args, hf_token)

    # Проверим, что файлы существуют, до запуска тяжёлых моделей
    files = [Path(f) for f in args.files]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        log.error("Файлы не найдены: %s", ", ".join(missing))
        return 1

    pipeline = AnalysisPipeline(config)

    any_failure = False
    for f in files:
        try:
            result = pipeline.run(str(f))
            print_result(result)
            if result.analysis is None or result.stages_failed:
                any_failure = True
        except KeyboardInterrupt:
            log.warning("Прервано пользователем.")
            return 130
        except Exception as exc:
            log.exception("Необработанная ошибка на файле %s: %s", f.name, exc)
            any_failure = True

    return 2 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())