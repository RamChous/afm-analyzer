"""
scripts/smoke_speech.py
Сквозной прогон транскрипции + диаризации на реальном файле.

Запуск:
    export HF_TOKEN="hf_..."
    python -m scripts.smoke_speech path/to/video.mp4 --language ru --speakers 3
"""

import os
import sys
import json
import time
import argparse
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from src.speech.transcription import Transcriber
from src.speech.diarization import Diarizer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: transcription + diarization")
    parser.add_argument("file", help="Путь к видео или аудио")
    parser.add_argument("--language", default=None, help="Код языка, напр. 'ru'. Иначе авто.")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Подсказка по числу говорящих (используется как потолок)")
    parser.add_argument("--model", default="large-v3", help="Размер модели Whisper")
    parser.add_argument("--out", default="out_speech.json", help="Куда сохранить результат")
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        log.error("Файл не найден: %s", src)
        return 1

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.error("Не задан HF_TOKEN (export HF_TOKEN=hf_...)")
        return 1

    # --- Извлекаем аудио ОДИН раз ------------------------------------
    tmp_wav = None
    if src.suffix.lower() in VIDEO_EXTS:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        log.info("Извлекаю аудио → %s", tmp_wav)
        Transcriber._extract_audio(str(src), tmp_wav)
        audio_for_pipeline = tmp_wav
    else:
        audio_for_pipeline = str(src)

    try:
        # --- Транскрипция --------------------------------------------
        t0 = time.perf_counter()
        transcriber = Transcriber(model_size=args.model, language=args.language)
        transcription = transcriber.transcribe(audio_for_pipeline)
        t_transcribe = time.perf_counter() - t0

        words_total = sum(len(s.words) for s in transcription.segments)
        log.info("Транскрипция: %d сегментов, %d слов, язык=%s, %.1fs аудио, заняло %.1fs",
                 len(transcription.segments), words_total,
                 transcription.language, transcription.duration or 0.0, t_transcribe)

        if words_total == 0:
            log.warning("НЕТ word-таймстампов! Диаризация уйдёт в посегментный режим. "
                        "Проверь, что word_timestamps=True.")

        # --- Диаризация (переиспользуем тот же wav) ------------------
        t0 = time.perf_counter()
        diarizer = Diarizer(hf_token=hf_token)
        diarization = diarizer.diarize(
            file_path=str(src),
            transcription=transcription,
            speaker_count=args.speakers,
            audio_path=audio_for_pipeline,   # ← повторного извлечения не будет
        )
        t_diarize = time.perf_counter() - t0
        log.info("Диаризация: %d спикеров, %d реплик, заняло %.1fs",
                 diarization.num_speakers, len(diarization.utterances), t_diarize)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    # --- Печать диалога ----------------------------------------------
    print("\n" + "=" * 70)
    print("РАСШИФРОВКА ПО СПИКЕРАМ")
    print("=" * 70)
    for utt in diarization.utterances:
        print(f"[{fmt_ts(utt.start)}–{fmt_ts(utt.end)}] "
              f"{utt.speaker:<11} (conf {utt.confidence:.2f}): {utt.text}")

    # --- Простые проверки целостности --------------------------------
    print("\n" + "-" * 70)
    print("ПРОВЕРКИ")
    print("-" * 70)
    checks = []
    checks.append(("Есть реплики", len(diarization.utterances) > 0))
    checks.append(("Найден хотя бы 1 спикер", diarization.num_speakers >= 1))
    unknown = sum(1 for u in diarization.utterances if u.speaker == "UNKNOWN")
    checks.append(("UNKNOWN-реплик < 30%",
                   unknown <= 0.3 * max(len(diarization.utterances), 1)))
    monotonic = all(
        diarization.utterances[i].start <= diarization.utterances[i + 1].start
        for i in range(len(diarization.utterances) - 1)
    )
    checks.append(("Реплики идут по времени", monotonic))

    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if unknown:
        print(f"  (UNKNOWN-реплик: {unknown})")

    # --- Сохраняем для инспекции / кэша ------------------------------
    out = Path(args.out)
    out.write_text(json.dumps({
        "transcription": transcription.to_dict(),
        "diarization": diarization.to_dict(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Результат сохранён → %s", out)

    print("\n--- RAW SPEAKER TURNS ---")
    for t in diarization.speaker_turns:
        print(f"  {t.speaker:<12} {t.start:6.2f} – {t.end:6.2f}  ({t.end - t.start:.2f}s)")
    print(f"num_speakers = {diarization.num_speakers}")

    return 0 if all(ok for _, ok in checks) else 2


if __name__ == "__main__":
    sys.exit(main())

