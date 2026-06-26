"""
tests/test_rules_parser.py
Тесты парсера search_procedures.txt.
Запуск: pytest tests/test_rules_parser.py -v
"""

from pathlib import Path

import pytest

from src.llm.rules import parse_rules, rules_index
from src.models.llm import Severity


# ---------------------------------------------------------------------------
# Часть 1. Синтетический файл — детерминированная проверка логики парсера
# ---------------------------------------------------------------------------

SAMPLE = """\
# комментарий-шапка, должен игнорироваться
# ещё строка комментария

## СТАДИЯ: НАЧАЛО ОБЫСКА

[R01] severity=high check=audio
Следователь обязан представиться сам и представить
всех участников следственного действия.

[R02] severity=critical check=audio
Следователь обязан предъявить постановление о производстве обыска.

## СТАДИЯ: ФИКСАЦИЯ

[R10] severity=medium check=video+audio
Видеосъёмка должна начинаться с момента прибытия.
"""


@pytest.fixture
def sample_file(tmp_path) -> str:
    """Пишет синтетический файл правил во временную папку."""
    p = tmp_path / "rules.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    return str(p)


def test_parses_all_rules(sample_file):
    rules = parse_rules(sample_file)
    assert len(rules) == 3
    assert [r.rule_id for r in rules] == ["R01", "R02", "R10"]


def test_severity_parsed_correctly(sample_file):
    idx = rules_index(parse_rules(sample_file))
    assert idx["R01"].severity == Severity.HIGH
    assert idx["R02"].severity == Severity.CRITICAL
    assert idx["R10"].severity == Severity.MEDIUM


def test_check_field_parsed(sample_file):
    idx = rules_index(parse_rules(sample_file))
    assert idx["R01"].check == "audio"
    assert idx["R10"].check == "video+audio"   # составной check не ломается


def test_stage_assigned_from_section(sample_file):
    idx = rules_index(parse_rules(sample_file))
    assert idx["R01"].stage == "НАЧАЛО ОБЫСКА"
    assert idx["R02"].stage == "НАЧАЛО ОБЫСКА"
    assert idx["R10"].stage == "ФИКСАЦИЯ"        # стадия сменилась


def test_multiline_text_joined(sample_file):
    """Текст правила из нескольких строк склеивается в одну."""
    idx = rules_index(parse_rules(sample_file))
    assert idx["R01"].text == (
        "Следователь обязан представиться сам и представить "
        "всех участников следственного действия."
    )


def test_comments_ignored(sample_file):
    """Строки-комментарии (#) не превращаются в правила."""
    rules = parse_rules(sample_file)
    # Если бы комментарии парсились — правил было бы больше трёх
    assert all(r.rule_id.startswith("R") for r in rules)


def test_unknown_severity_skipped(tmp_path):
    """Правило с неизвестным severity пропускается, а не роняет парсер."""
    p = tmp_path / "bad.txt"
    p.write_text(
        "## СТАДИЯ: ТЕСТ\n"
        "[R01] severity=high check=audio\nОК правило.\n\n"
        "[R99] severity=ОГРОМНЫЙ check=audio\nКривое правило.\n",
        encoding="utf-8",
    )
    rules = parse_rules(str(p))
    ids = [r.rule_id for r in rules]
    assert ids == ["R01"]          # R99 с битым severity отброшено


def test_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("# только комментарий\n\n", encoding="utf-8")
    assert parse_rules(str(p)) == []


def test_rules_index_maps_by_id(sample_file):
    idx = rules_index(parse_rules(sample_file))
    assert set(idx.keys()) == {"R01", "R02", "R10"}
    assert idx["R02"].rule_id == "R02"


# ---------------------------------------------------------------------------
# Часть 2. Реальный файл — ловит опечатки прямо в search_procedures.txt
# ---------------------------------------------------------------------------

# Ищем реальный файл в нескольких типичных местах. Если не найден — тесты
# части 2 просто пропускаются (skip), а не падают.
_CANDIDATES = [
    Path("config/procedures/search_procedures.txt"),
    Path("afm_analyzer_main/config/procedures/search_procedures.txt"),
]
_REAL = next((p for p in _CANDIDATES if p.exists()), None)
_real_only = pytest.mark.skipif(_REAL is None, reason="search_procedures.txt не найден")


@_real_only
def test_real_file_rule_count():
    """В реальном файле ровно 16 правил (R01–R05, R10–R13, R20–R21, R30–R32, R40–R41)."""
    rules = parse_rules(str(_REAL))
    assert len(rules) == 16, f"Ожидали 16 правил, распарсилось {len(rules)}"


@_real_only
def test_real_file_ids_unique():
    """Нет дублей rule_id — частая опечатка при ручном редактировании."""
    rules = parse_rules(str(_REAL))
    ids = [r.rule_id for r in rules]
    assert len(ids) == len(set(ids)), f"Дубли rule_id: {ids}"


@_real_only
def test_real_file_all_have_text():
    """У каждого правила есть непустой текст."""
    rules = parse_rules(str(_REAL))
    empty = [r.rule_id for r in rules if not r.text.strip()]
    assert not empty, f"Правила без текста: {empty}"


@_real_only
def test_real_file_known_severities():
    """Все severity — из допустимого множества (Severity их уже провалидировал)."""
    rules = parse_rules(str(_REAL))
    # Точечная проверка нескольких критичных, как мы их утвердили
    idx = rules_index(rules)
    assert idx["R02"].severity == Severity.CRITICAL
    assert idx["R04"].severity == Severity.CRITICAL
    assert idx["R13"].severity == Severity.CRITICAL
    assert idx["R20"].severity == Severity.CRITICAL
    assert idx["R31"].severity == Severity.CRITICAL
    assert idx["R40"].severity == Severity.LOW