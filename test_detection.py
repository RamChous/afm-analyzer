"""
test_detection.py
Запускает анализ видео + речи и выводит единый результат:
- сколько участников на видео (YOLO)
- сколько уникальных голосов (pyannote)
- объединённый вывод с доказательной базой
"""

import os
from dotenv import load_dotenv

from src.video.frame_analysis import analyze_frames
from src.video.person_detection import detect_persons
from src.speech.transcription import transcribe_video
from src.speech.diarization import diarize_video

load_dotenv()
hf_token = os.getenv("hf_token")
VIDEO_PATH = "test2.mp4"

# ───────────────────────────────────────────────
# 1. Анализ видео
# ───────────────────────────────────────────────
print("=" * 50)
print("ШАГ 1: Анализ видео (YOLO)")
print("=" * 50)

video_result = analyze_frames(VIDEO_PATH, sample_rate=1.0)
persons = detect_persons(video_result)

print(f"Макс. людей в кадре одновременно : {persons.max_count}")
print(f"Момент пика                       : {persons.peak_timestamp:.1f}с")

# ───────────────────────────────────────────────
# 2. Транскрипция
# ───────────────────────────────────────────────
print()
print("=" * 50)
print("ШАГ 2: Транскрипция (Whisper)")
print("=" * 50)

transcription = transcribe_video(VIDEO_PATH)

print(f"Язык                              : {transcription.language}")
print(f"Длительность                      : {transcription.duration:.1f}с")
print(f"Текст (первые 300 символов)       :")
print(f"  {transcription.full_text}...")

# ───────────────────────────────────────────────
# 3. Диаризация — кто говорит
# ───────────────────────────────────────────────
print()
print("=" * 50)
print("ШАГ 3: Диаризация (pyannote)")
print("=" * 50)

diarization = diarize_video(
    file_path=VIDEO_PATH,
    transcription=transcription,
    hf_token=hf_token,
    speaker_count=None,   # pyannote определит сам, потом сравним с YOLO
)

print(f"Уникальных голосов найдено        : {diarization.num_speakers}")
print()
for speaker in diarization.speakers:
    utterances = diarization.utterances_by_speaker(speaker)
    total_speech = sum(u.end - u.start for u in utterances)
    print(f"  {speaker}:")
    print(f"    Реплик       : {len(utterances)}")
    print(f"    Время речи   : {total_speech:.1f}с")
    print(f"    Пример       : \"{utterances[0].text[:100]}\"" if utterances else "")

# ───────────────────────────────────────────────
# 4. Объединённый вывод
# ───────────────────────────────────────────────
print()
print("=" * 50)
print("ИТОГ: Участники обыска")
print("=" * 50)

video_count  = persons.max_count
speech_count = diarization.num_speakers
final_count  = max(video_count, speech_count)

print(f"По видео (YOLO)                   : {video_count} чел. (пик на {persons.peak_timestamp:.1f}с)")
print(f"По голосу (pyannote)              : {speech_count} уникальных голоса/голосов")
print(f"Итоговое кол-во участников        : {final_count}")
print()
print("Участники и их речь:")
print("-" * 50)
for i, speaker in enumerate(diarization.speakers, 1):
    utterances = diarization.utterances_by_speaker(speaker)
    total_speech = sum(u.end - u.start for u in utterances)
    print(f"СПИКЕР {i} [{speaker}] — {total_speech:.1f}с речи")
    # Показываем первые 3 реплики как пример
    for utt in utterances[:3]:
        print(f"  [{utt.start:.1f}с] \"{utt.text}\"")
    if len(utterances) > 3:
        print(f"  ... ещё {len(utterances) - 3} реплик")
    print()

# Проверка согласованности
print("-" * 50)
if video_count == speech_count:
    print(f"✓ Данные согласованы: видео и голос подтверждают {final_count} участников")
elif speech_count > video_count:
    print(f"⚠ Голосов больше чем людей в кадре — возможно кто-то за камерой")
else:
    print(f"⚠ Людей в кадре больше чем голосов — возможно не все говорили")