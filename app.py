from src.speech.transcription import transcribe_video, Transcriber

result = transcribe_video("test2.mp4", model_size="large-v3")
print(result.full_text)