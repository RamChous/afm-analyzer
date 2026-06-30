import os
import socket

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def check_env() -> None:
    if not os.environ.get("HF_TOKEN"):
        print("  ⚠ HF_TOKEN не задан в .env — диаризация говорящих не будет работать.")
    print(f"  Модель LLM: {os.environ.get('LLM_MODEL', 'qwen3:14b')} (Ollama, локально)")
    print(f"  Воркеров пайплайна: {os.environ.get('PIPELINE_MAX_WORKERS', '2')}")


if __name__ == "__main__":
    local_ip = get_local_ip()

    print("-" * 56)
    print("  Inspectra — АФМ РК · Анализ обысков")
    print("-" * 56)
    print(f"  Локально:  http://localhost:8000")
    print(f"  По сети:   http://{local_ip}:8000")
    print(f"  Тест:      admin / admin123")
    print("-" * 56)
    check_env()
    print("  Для остановки: Ctrl+C")

    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=bool(os.environ.get("DEV_RELOAD", "")),
    )