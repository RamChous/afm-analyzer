"""
src/reports/excel_export.py
Выгрузка AnalysisResult в Excel — отчёт для человека (следователь/проверяющий).
Два листа: Сводка и Нарушения. Строки нарушений окрашены по критичности.
"""

import logging
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.models.llm import AnalysisResult, Severity

logger = logging.getLogger(__name__)

# Цвета заливки по критичности (наша разметка, не финансовая)
SEVERITY_FILL = {
    Severity.CRITICAL: "C00000",   # тёмно-красный
    Severity.HIGH:     "FF6D6D",   # красный
    Severity.MEDIUM:   "FFC000",   # оранжевый
    Severity.LOW:      "FFE699",   # жёлтый
}
SEVERITY_LABEL = {
    Severity.CRITICAL: "Критическое",
    Severity.HIGH:     "Высокое",
    Severity.MEDIUM:   "Среднее",
    Severity.LOW:      "Низкое",
}

FONT = "Arial"
HEADER_FILL = "1F3864"     # тёмно-синий заголовок
HEADER_FONT_COLOR = "FFFFFF"
_THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def export_excel(result: AnalysisResult, output_path: str) -> str:
    """
    Сохраняет результат анализа в .xlsx с двумя листами.

    Returns:
        Путь к сохранённому файлу.
    """
    wb = Workbook()
    _build_summary_sheet(wb.active, result)
    _build_violations_sheet(wb.create_sheet("Нарушения"), result)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    logger.info("Excel-отчёт сохранён: %s (%d нарушений)",
                path, len(result.violations))
    return str(path)


# ---------------------------------------------------------------------------
# Лист "Сводка"
# ---------------------------------------------------------------------------

def _build_summary_sheet(ws, result: AnalysisResult) -> None:
    ws.title = "Сводка"
    ws.sheet_view.showGridLines = False

    ws["A1"] = "ОТЧЁТ ОБ АНАЛИЗЕ ВИДЕОЗАПИСИ ОБЫСКА"
    ws["A1"].font = Font(name=FONT, size=14, bold=True)
    ws.merge_cells("A1:B1")

    rows = [
        ("Файл:", result.file_path),
        ("Сгенерирован:", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Модель:", result.model_used),
        ("Итог:", result.summary),
        ("", ""),
        ("РОЛИ ГОВОРЯЩИХ", ""),
    ]
    for speaker, role in (result.speaker_roles or {}).items():
        rows.append((speaker, role))
    if not result.speaker_roles:
        rows.append(("—", "роли не определены"))

    rows += [
        ("", ""),
        ("НАРУШЕНИЯ ПО КРИТИЧНОСТИ", ""),
    ]
    counts = _counts_by_severity(result)
    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
        rows.append((SEVERITY_LABEL[sev], counts.get(sev, 0)))

    r = 3
    for label, value in rows:
        ws.cell(row=r, column=1, value=label).font = Font(name=FONT, bold=bool(
            label and label.isupper()))
        ws.cell(row=r, column=2, value=value).font = Font(name=FONT)
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        r += 1

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 70


def _counts_by_severity(result: AnalysisResult) -> dict:
    counts: dict = {}
    for v in result.violations:
        counts[v.severity] = counts.get(v.severity, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Лист "Нарушения"
# ---------------------------------------------------------------------------

HEADERS = [
    ("Время", 10),
    ("Критичность", 14),
    ("Правило", 10),
    ("Требование", 45),
    ("Цитата (доказательство)", 45),
    ("Обоснование", 45),
    ("Уверенность", 12),
    ("Стадия", 22),
]


def _build_violations_sheet(ws, result: AnalysisResult) -> None:
    ws.sheet_view.showGridLines = False

    # Заголовок таблицы
    for col, (title, width) in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=title)
        cell.font = Font(name=FONT, bold=True, color=HEADER_FONT_COLOR)
        cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(col)].width = width

    ws.freeze_panes = "A2"   # шапка не уезжает при прокрутке

    if not result.violations:
        ws.cell(row=2, column=1, value="Нарушений не выявлено").font = Font(
            name=FONT, italic=True)
        ws.merge_cells("A2:H2")
        return

    for i, v in enumerate(result.violations_by_severity(), start=2):
        values = [
            _fmt_ts(v.timestamp),
            SEVERITY_LABEL[v.severity],
            v.rule_id,
            v.rule_text,
            v.quote,
            v.reasoning,
            f"{v.confidence:.0%}",
            v.stage,
        ]
        fill = PatternFill("solid", fgColor=SEVERITY_FILL[v.severity])
        # Тёмные заливки (critical/high) — белый текст для контраста
        light_text = v.severity in (Severity.CRITICAL, Severity.HIGH)
        for col, val in enumerate(values, start=1):
            cell = ws.cell(row=i, column=col, value=val)
            cell.font = Font(name=FONT,
                             color="FFFFFF" if (light_text and col == 2) else "000000")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = BORDER
            # Заливкой выделяем только колонку "Критичность" — чтобы таблица
            # читалась, но не рябила целиком
            if col == 2:
                cell.fill = fill