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
print(f"Текст:")
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

def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


print()
print("=" * 80)
print("ИТОГ: Участники обыска")
print("=" * 80)

video_count = persons.max_count
speech_count = diarization.num_speakers
final_count = max(video_count, speech_count)

print(
    f"По видео (YOLO): {video_count} чел. "
    f"(пик на {format_time(persons.peak_timestamp)})"
)

print(
    f"По голосу (pyannote): "
    f"{speech_count} уникальных голосов"
)

print(
    f"Итоговое количество участников: "
    f"{final_count}"
)

print()
print("=" * 80)
print("ПОЛНАЯ РЕЧЬ ПО СПИКЕРАМ")
print("=" * 80)

for i, speaker in enumerate(diarization.speakers, 1):

    utterances = diarization.utterances_by_speaker(speaker)

    total_speech = sum(
        u.end - u.start
        for u in utterances
    )

    print()
    print("=" * 80)
    print(
        f"СПИКЕР {i} [{speaker}] "
        f"(время речи: {total_speech:.1f} сек)"
    )
    print("=" * 80)

    for utt in utterances:

        start_time = format_time(utt.start)
        end_time = format_time(utt.end)

        print(
            f"[{start_time} - {end_time}] "
            f"{utt.text}"
        )

print()
print("-" * 80)

if video_count == speech_count:
    print(
        f"✓ Данные согласованы: "
        f"{final_count} участников"
    )

elif speech_count > video_count:
    print(
        "⚠ Голосов больше, чем людей в кадре "
        "(возможно кто-то за камерой)"
    )

else:
    print(
        "⚠ Людей в кадре больше, чем голосов "
        "(возможно не все говорили)"
    )