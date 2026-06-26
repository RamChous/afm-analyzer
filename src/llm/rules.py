"""Парсер search_procedures.txt → список Rule."""

import re
import logging
from pathlib import Path

from src.models.llm import Rule, Severity

logger = logging.getLogger(__name__)

# [R04] severity=critical check=audio
_HEADER_RE = re.compile(
    r"^\[(?P<id>R\d+)\]\s+severity=(?P<sev>\w+)\s+check=(?P<check>[\w+]+)\s*$"
)
_STAGE_RE = re.compile(r"^##\s*СТАДИЯ:\s*(?P<stage>.+?)\s*$", re.IGNORECASE)


def parse_rules(path: str) -> list[Rule]:
    """
    Парсит файл правил. Формат:
        ## СТАДИЯ: НАЧАЛО ОБЫСКА
        [R01] severity=high check=audio
        Текст требования, возможно
        в несколько строк.
    Пустые строки и строки, начинающиеся с '#' (но не '##'), игнорируются.
    """
    rules: list[Rule] = []
    current_stage = ""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        stage_m = _STAGE_RE.match(line)
        if stage_m:
            current_stage = stage_m.group("stage").strip()
            i += 1
            continue

        header_m = _HEADER_RE.match(line)
        if header_m:
            # Собираем текст до следующего заголовка/секции/пустого блока
            text_lines: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i].rstrip()
                if (not nxt) or _HEADER_RE.match(nxt) or _STAGE_RE.match(nxt) \
                        or nxt.startswith("#"):
                    break
                text_lines.append(nxt.strip())
                i += 1

            try:
                severity = Severity(header_m.group("sev").lower())
            except ValueError:
                logger.warning("Неизвестный severity '%s' в %s — пропускаю",
                               header_m.group("sev"), header_m.group("id"))
                continue

            rules.append(Rule(
                rule_id=header_m.group("id"),
                severity=severity,
                check=header_m.group("check"),
                text=" ".join(text_lines).strip(),
                stage=current_stage,
            ))
            continue

        i += 1

    logger.info("Загружено правил: %d", len(rules))
    return rules


def rules_index(rules: list[Rule]) -> dict[str, Rule]:
    """Словарь rule_id → Rule для быстрой подстановки severity."""
    return {r.rule_id: r for r in rules}