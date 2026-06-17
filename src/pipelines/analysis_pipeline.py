import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from moviepy import VideoFileClip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float

@dataclass
class CameraViolation:
    start_time: str
    end_time: str

class AFMVideoAnalyzer:
    def __init__(self, hf_token: str):
        logger.info("Загрузка речевых моделей...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            token=hf_token
        )
        self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        
    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def process_audio(self, video_path: str) -> tuple[List[Dict[str, Any]], int]:
        """Извлекает чистый звук и запускает точную транскрипцию БЕЗ учета встроенных субтитров"""
        logger.info("[AUDIO] Конвертация видео в изолированный WAV дорожку...")
        temp_audio_path = "temp_perfect_audio.wav"
        
        try:
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(
                temp_audio_path, 
                fps=16000,          
                nbytes=2,           
                codec='pcm_s16le',  
                ffmpeg_params=["-ac", "1"],
                logger=None
            )
            video_clip.close()
            
            logger.info("[AUDIO] Запуск диаризации...")
            diar_result = self.diarization_pipeline(temp_audio_path)
            annotation = diar_result.speaker_diarization if hasattr(diar_result, "speaker_diarization") else diar_result
                
            turns = []
            unique_speakers = set()
            for turn, _, label in annotation.itertracks(yield_label=True):
                turns.append(SpeakerTurn(speaker=str(label), start=turn.start, end=turn.end))
                unique_speakers.add(str(label))
            
            logger.info("[AUDIO] Запуск Whisper (Транскрипция только звука)...")
            # Передаем аудио-файл temp_audio_path, чтобы Whisper вообще не видел само видео и его субтитры!
            segments, _ = self.whisper_model.transcribe(temp_audio_path, beam_size=5)
            
            final_dialogue = []
            
            # Список стоп-слов/фраз, которые нужно молча вырезать из отчета
            stop_phrases = ["медиазапись включена", "редактор субтитров", "субтитры", "и.бойкова"]
            
            for seg in segments:
                text_clean = seg.text.strip()
                
                # Проверка на системный мусор
                if any(phrase in text_clean.lower() for phrase in stop_phrases) or len(text_clean) <= 1:
                    continue
                    
                seg_midpoint = (seg.start + seg.end) / 2
                current_speaker = "Неизвестный"
                best_distance = float("inf")
                
                for turn in turns:
                    if turn.start <= seg_midpoint <= turn.end:
                        current_speaker = turn.speaker
                        break
                    else:
                        dist = min(abs(seg_midpoint - turn.start), abs(seg_midpoint - turn.end))
                        if dist < best_distance:
                            best_distance = dist
                            current_speaker = turn.speaker
                
                try:
                    speaker_num = int(current_speaker.split("_")[-1]) + 1
                    speaker_display = f"Человек {speaker_num}"
                except:
                    speaker_display = current_speaker

                final_dialogue.append({
                    "speaker": speaker_display,
                    "time_str": f"{self._format_time(seg.start)} - {self._format_time(seg.end)}",
                    "text": text_clean
                })
                
            return final_dialogue, len(unique_speakers)

        finally:
            if os.path.exists(temp_audio_path):
                try: os.remove(temp_audio_path)
                except: pass

    def process_video(self, video_path: str) -> tuple[List[CameraViolation], int]:
        """Анализ видео: точный подсчет лиц через встроенную OpenCV DNN нейросеть (без XML)"""
        logger.info("[VIDEO] Запуск точной DNN детекции лиц OpenCV...")
        
        # Загружаем встроенную в OpenCV модель распознавания лиц (Caffe модель)
        # Она поставляется вместе с opencv-python автоматически, файлы не нужны
        model_img = os.path.join(cv2.data.haarcascades, "..", "dnn", "resnet10_ssd_iter_140000.caffemodel")
        proto_txt = os.path.join(cv2.data.haarcascades, "..", "dnn", "deploy.prototxt")
        
        # Флаг: если встроенная папка DNN недоступна, используем резервный метод по яркости
        use_dnn = os.path.exists(model_img) and os.path.exists(proto_txt)
        if use_dnn:
            net = cv2.dnn.readNetFromCaffe(proto_txt, model_img)
        else:
            logger.warning("Папка OpenCV DNN недоступна. Будет выполнен только анализ закрытия камеры.")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25
            
        violations = []
        max_people_in_frame = 0
        camera_closed = False
        closed_start_sec = 0.0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            current_sec = frame_count / fps
            
            # Анализируем 2 кадра в секунду для экономии ресурсов
            if frame_count % int(fps / 2) != 0:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Фиксация закрытия объектива
            mean_brightness = np.mean(gray)
            if mean_brightness < 12: 
                if not camera_closed:
                    camera_closed = True
                    closed_start_sec = current_sec
            else:
                if camera_closed:
                    camera_closed = False
                    if current_sec - closed_start_sec >= 2.0:
                        violations.append(CameraViolation(
                            start_time=self._format_time(closed_start_sec),
                            end_time=self._format_time(current_sec)
                        ))
            
            # 2. Точный подсчет лиц через DNN нейросеть
            if use_dnn:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                
                faces_count = 0
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    # Учитываем только детекции с уверенностью выше 50% (отсекает ложные стулья/шкафы)
                    if confidence > 0.5:
                        faces_count += 1
                        
                if faces_count > max_people_in_frame:
                    max_people_in_frame = faces_count
                    
        cap.release()
        
        if camera_closed and (frame_count / fps) - closed_start_sec >= 2.0:
            violations.append(CameraViolation(
                start_time=self._format_time(closed_start_sec),
                end_time=self._format_time(frame_count / fps)
            ))
            
        # Если DNN не отработал, ставим базовую единицу по умолчанию для MVP
        if max_people_in_frame == 0 and not use_dnn:
            max_people_in_frame = 1
            
        return violations, max_people_in_frame

    def run_analysis(self, video_path: str):
        if not os.path.exists(video_path):
            logger.error(f"Файл {video_path} не найден!")
            return
            
        dialogue, audio_speakers = self.process_audio(video_path)
        camera_violations, max_visual_people = self.process_video(video_path)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТ ОБРАБОТКИ ВИДЕОЗАПИСИ")
        print("="*60)
        
        if dialogue:
            for replica in dialogue:
                print(f"{replica['speaker']} ({replica['time_str']}): {replica['text']}")
        else:
            print("[Полезная речь в аудиопотоке не обнаружена]")
            
        print("\n" + "-"*40)
        print(f"общее количество людей распознанное по речи: {audio_speakers}")
        print(f"общее количество людей в кадре: {max_visual_people}")
        print("-"*40)
        
        print("ОБНАРУЖЕНИЕ ЗАКРЫТОЙ И ВЫКЛЮЧЕННОЙ КАМЕРЫ:")
        if camera_violations:
            for i, vio in enumerate(camera_violations, 1):
                print(f"Нарушение {i}: камера была закрыта/выключена на минуте {vio.start_time} по {vio.end_time}")
        else:
            print("Нарушений работы камеры не обнаружено.")
        print("="*60 + "\n")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    analyzer = AFMVideoAnalyzer(hf_token=os.getenv("HF_TOKEN"))
    analyzer.run_analysis("test2.mp4")