"""
Тонкий клиент к локальной Ollama для structured-output запросов.
Изолирует движок: остальной код не знает, что под капотом Ollama.
"""

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class LLMClient:
    """
    Usage:
        client = LLMClient(model="qwen3:14b")
        data = client.generate_json(
            system="Ты юрист-аналитик...",
            prompt="Проанализируй...",
            schema=VIOLATIONS_SCHEMA,
        )
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        temperature: float = 0.0,
        timeout: int = 600,
        max_retries: int = 2,
    ):
        """
        Args:
            model:       имя модели в Ollama (qwen3:30b и т.п.) — из конфига, не хардкод.
            num_ctx:     контекстное окно. Ollama по умолчанию режет до 2048 —
                         поэтому задаём явно.
            temperature: 0.0 для детерминированного юридического анализа.
            timeout:     секунды; на CPU генерация долгая.
            max_retries: повторы при невалидном JSON.
        """
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        logger.info("LLMClient init (model=%s, num_ctx=%d, host=%s)",
                    model, num_ctx, host)

    def generate_json(
        self,
        system: str,
        prompt: str,
        schema: dict,
    ) -> dict:
        """
        Запрос с принудительным JSON-выводом по schema.
        Ollama поддерживает 'format' = JSON Schema — движок сам форсит форму.
        Возвращает распарсенный dict. Бросает OllamaError при неудаче.
        """
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "format": schema,           # structured output по JSON Schema
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                return json.loads(raw)          # format=schema → это чистый JSON
            except json.JSONDecodeError as exc:
                last_err = exc
                logger.warning("Попытка %d: модель вернула невалидный JSON: %s",
                               attempt, exc)
            except requests.RequestException as exc:
                last_err = exc
                logger.warning("Попытка %d: ошибка запроса к Ollama: %s",
                               attempt, exc)

        raise OllamaError(
            f"Не удалось получить валидный JSON за {self.max_retries + 1} попыток: {last_err}"
        )

    def health_check(self) -> bool:
        """Проверка, что Ollama жива и модель доступна."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                logger.warning("Модель %s не найдена в Ollama. Доступны: %s",
                               self.model, models)
                return False
            return True
        except requests.RequestException as exc:
            logger.error("Ollama недоступна на %s: %s", self.host, exc)
            return False