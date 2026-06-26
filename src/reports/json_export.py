"""
src/reports/json_export.py
Выгрузка AnalysisResult в JSON — машиночитаемый отчёт для архива/интеграции.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models.llm import AnalysisResult

logger = logging.getLogger(__name__)

REPORT_VERSION = "1.0"


def export_json(
    result: AnalysisResult,
    output_path: str,
    extra: Optional[dict] = None,
) -> str:
    """
    Сохраняет результат анализа в JSON.

    Args:
        result:      результат LLM-анализа.
        output_path: куда сохранить (.json).
        extra:       опциональные доп. данные (например, сырые
                     транскрипция/диаризация/обструкции для полного архива).

    Returns:
        Путь к сохранённому файлу.
    """
    payload = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": result.to_dict(),
    }
    if extra:
        payload["raw"] = extra

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSON-отчёт сохранён: %s (%d нарушений)",
                path, len(result.violations))
    return str(path)