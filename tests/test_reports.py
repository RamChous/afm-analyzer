"""
tests/test_reports.py
Тесты экспортёров JSON и Excel.
Запуск: pytest tests/test_reports.py -v
"""

import json

import pytest
from openpyxl import load_workbook

from src.reports.json_export import export_json, REPORT_VERSION
from src.reports.excel_export import export_excel, SEVERITY_LABEL
from src.models.llm import AnalysisResult, Violation, Severity


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

def violation(rule_id, sev, ts, quote="цитата", conf=0.8):
    return Violation(
        rule_id=rule_id,
        rule_text="текст требования",
        severity=sev,
        timestamp=ts,
        quote=quote,
        reasoning="обоснование",
        confidence=conf,
        stage="ТЕСТ",
    )


@pytest.fixture
def result_with_violations():
    return AnalysisResult(
        file_path="video.mp4",
        speaker_roles={"SPEAKER_00": "следователь", "SPEAKER_01": "обыскиваемый"},
        violations=[
            violation("R04", Severity.CRITICAL, 330.0, quote="Начинаем осмотр"),
            violation("R01", Severity.HIGH, 12.0),
            violation("R40", Severity.LOW, 600.0),
        ],
        summary="Выявлено нарушений: 3 (1 критических, 1 высоких, 1 низких).",
        model_used="qwen3:30b",
    )


@pytest.fixture
def result_empty():
    return AnalysisResult(
        file_path="clean.mp4",
        speaker_roles={},
        violations=[],
        summary="Нарушений по проверяемым требованиям не выявлено.",
        model_used="qwen3:30b",
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def test_json_file_created(tmp_path, result_with_violations):
    out = tmp_path / "report.json"
    path = export_json(result_with_violations, str(out))
    assert out.exists()
    assert path == str(out)


def test_json_valid_and_has_metadata(tmp_path, result_with_violations):
    out = tmp_path / "report.json"
    export_json(result_with_violations, str(out))
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["report_version"] == REPORT_VERSION
    assert "generated_at" in data
    assert "analysis" in data


def test_json_contains_violations(tmp_path, result_with_violations):
    out = tmp_path / "report.json"
    export_json(result_with_violations, str(out))
    analysis = json.loads(out.read_text(encoding="utf-8"))["analysis"]

    assert analysis["violation_count"] == 3
    ids = [v["rule_id"] for v in analysis["violations"]]
    assert set(ids) == {"R04", "R01", "R40"}


def test_json_violations_sorted_by_severity(tmp_path, result_with_violations):
    """В отчёте нарушения идут от critical к low (через violations_by_severity)."""
    out = tmp_path / "report.json"
    export_json(result_with_violations, str(out))
    violations = json.loads(out.read_text(encoding="utf-8"))["analysis"]["violations"]
    severities = [v["severity"] for v in violations]
    assert severities == ["critical", "high", "low"]


def test_json_preserves_cyrillic(tmp_path, result_with_violations):
    """Кириллица сохраняется как текст, а не \\uXXXX."""
    out = tmp_path / "report.json"
    export_json(result_with_violations, str(out))
    raw = out.read_text(encoding="utf-8")
    assert "следователь" in raw          # ensure_ascii=False работает


def test_json_extra_included(tmp_path, result_empty):
    out = tmp_path / "report.json"
    export_json(result_empty, str(out), extra={"transcription": {"x": 1}})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["raw"]["transcription"]["x"] == 1


def test_json_creates_parent_dirs(tmp_path, result_empty):
    out = tmp_path / "nested" / "deep" / "report.json"
    export_json(result_empty, str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def test_excel_file_created(tmp_path, result_with_violations):
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    assert out.exists()


def test_excel_has_two_sheets(tmp_path, result_with_violations):
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    wb = load_workbook(out)
    assert "Сводка" in wb.sheetnames
    assert "Нарушения" in wb.sheetnames


def test_excel_violation_rows_match_count(tmp_path, result_with_violations):
    """Строк с нарушениями = число нарушений (+1 на шапку)."""
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    ws = load_workbook(out)["Нарушения"]

    # Считаем непустые строки в колонке "Время" (A), кроме шапки
    data_rows = [c for c in ws["A"][1:] if c.value]
    assert len(data_rows) == 3


def test_excel_header_row(tmp_path, result_with_violations):
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    ws = load_workbook(out)["Нарушения"]
    headers = [ws.cell(row=1, column=c).value for c in range(1, 9)]
    assert headers[0] == "Время"
    assert headers[2] == "Правило"
    assert "Цитата" in headers[4]


def test_excel_severity_label_present(tmp_path, result_with_violations):
    """В колонке критичности — человекочитаемые метки, не enum."""
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    ws = load_workbook(out)["Нарушения"]
    labels = [ws.cell(row=r, column=2).value for r in range(2, 5)]
    # Первая строка — critical (сортировка), значит "Критическое"
    assert labels[0] == SEVERITY_LABEL[Severity.CRITICAL]
    assert all(lbl in SEVERITY_LABEL.values() for lbl in labels)


def test_excel_quote_preserved(tmp_path, result_with_violations):
    """Цитата-доказательство попадает в отчёт (ядро проверяемости)."""
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    ws = load_workbook(out)["Нарушения"]
    quotes = [ws.cell(row=r, column=5).value for r in range(2, 5)]
    assert "Начинаем осмотр" in quotes


def test_excel_timestamp_formatted(tmp_path, result_with_violations):
    """Время в формате MM:SS, а не сырые секунды."""
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    ws = load_workbook(out)["Нарушения"]
    times = [ws.cell(row=r, column=1).value for r in range(2, 5)]
    assert "05:30" in times          # 330 секунд → 05:30
    assert "00:12" in times          # 12 секунд


def test_excel_empty_case(tmp_path, result_empty):
    """Без нарушений — файл создаётся, есть сообщение, нет строк-данных."""
    out = tmp_path / "report.xlsx"
    export_excel(result_empty, str(out))
    ws = load_workbook(out)["Нарушения"]
    assert ws.cell(row=2, column=1).value == "Нарушений не выявлено"


def test_excel_summary_sheet_has_roles(tmp_path, result_with_violations):
    """На листе Сводка присутствуют роли говорящих."""
    out = tmp_path / "report.xlsx"
    export_excel(result_with_violations, str(out))
    ws = load_workbook(out)["Сводка"]
    all_values = [c.value for row in ws.iter_rows() for c in row if c.value]
    joined = " ".join(str(v) for v in all_values)
    assert "следователь" in joined
    assert "SPEAKER_00" in joined