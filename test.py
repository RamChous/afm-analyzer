from src.speech.transcription import transcribe_video, Transcriber

result = transcribe_video("test2.mp4")
print(result.full_text)