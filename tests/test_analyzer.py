"""
tests/test_analyzer.py
Тесты оркестратора LLM-анализа — без реального Ollama.
Фейковый клиент возвращает заранее заданные ответы.
Запуск: pytest tests/test_analyzer.py -v
"""

import pytest

from src.llm.analyzer import Analyzer
from src.models.speech import Utterance, DiarizationResult
from src.models.llm import Rule, Severity


# ---------------------------------------------------------------------------
# Фейковый LLM-клиент
# ---------------------------------------------------------------------------

class FakeClient:
    """
    Подменяет LLMClient. Возвращает заготовленные ответы по очереди.
    roles_response — для прохода 1, violations_responses — список ответов
    на проходе 2 (по одному на чанк, по кругу если чанков больше).
    """
    def __init__(self, roles_response=None, violations_responses=None):
        self.model = "fake-model"
        self._roles = roles_response or {"roles": []}
        self._viol = violations_responses or [{"violations": []}]
        self.calls = []   # лог вызовов для проверок

    def generate_json(self, system, prompt, schema):
        self.calls.append({"system": system, "prompt": prompt})
        # Различаем проход по наличию слова в системном промпте
        if "роль" in system.lower() or "role" in system.lower():
            return self._roles
        # проход 2: отдаём ответы по очереди, последний повторяем
        idx = min(len([c for c in self.calls
                       if "ПРОВЕРЯЕМЫЕ" in c["prompt"]]) - 1,
                  len(self._viol) - 1)
        return self._viol[idx]


# ---------------------------------------------------------------------------
# Фикстуры данных
# ---------------------------------------------------------------------------

def rule(rid, sev, text="требование", check="audio", stage="ТЕСТ"):
    return Rule(rule_id=rid, severity=sev, check=check, text=text, stage=stage)


@pytest.fixture
def rules():
    return [
        rule("R01", Severity.HIGH, "представиться"),
        rule("R04", Severity.CRITICAL, "предложить добровольную выдачу"),
        rule("R40", Severity.LOW, "соблюдать этику"),
    ]


def utt(speaker, start, end, text="реплика"):
    return Utterance(speaker=speaker, start=start, end=end, text=text, confidence=0.9)


@pytest.fixture
def diar():
    return DiarizationResult(
        file_path="fake.mp4",
        num_speakers=2,
        utterances=[
            utt("SPEAKER_00", 0.0, 3.0, "Я следователь, предъявляю постановление"),
            utt("SPEAKER_01", 3.5, 5.0, "Понятно"),
            utt("SPEAKER_00", 5.5, 8.0, "Начинаем осмотр"),
        ],
    )


# ---------------------------------------------------------------------------
# Проход 3 (_assemble): severity из правила, не из модели
# ---------------------------------------------------------------------------

def test_severity_comes_from_rule_not_model(rules):
    """Даже если модель пришлёт severity — берётся из правила."""
    client = FakeClient(violations_responses=[{
        "violations": [{
            "rule_id": "R04",
            "timestamp": 5.5,
            "quote": "Начинаем осмотр",
            "reasoning": "не было предложения о выдаче",
            "confidence": 0.8,
            "severity": "low",   # модель врёт — должно игнорироваться
        }]
    }])
    analyzer = Analyzer(client, rules)
    out = analyzer._assemble(client._viol[0]["violations"])

    assert len(out) == 1
    assert out[0].severity == Severity.CRITICAL   # из правила R04, не "low"
    assert out[0].rule_text == "предложить добровольную выдачу"


def test_unknown_rule_id_dropped(rules):
    """Галлюцинация — несуществующий rule_id — отбрасывается."""
    analyzer = Analyzer(FakeClient(), rules)
    raw = [
        {"rule_id": "R99", "timestamp": 1.0, "quote": "x",
         "reasoning": "y", "confidence": 0.9},   # нет такого правила
        {"rule_id": "R01", "timestamp": 2.0, "quote": "x",
         "reasoning": "y", "confidence": 0.9},   # валидное
    ]
    out = analyzer._assemble(raw)
    assert [v.rule_id for v in out] == ["R01"]


def test_confidence_clamped(rules):
    """confidence вне [0,1] зажимается, нечисло → 0."""
    analyzer = Analyzer(FakeClient(), rules)
    raw = [
        {"rule_id": "R01", "timestamp": 1.0, "quote": "a",
         "reasoning": "b", "confidence": 1.7},
        {"rule_id": "R04", "timestamp": 2.0, "quote": "a",
         "reasoning": "b", "confidence": "nan-как-строка"},
    ]
    out = {v.rule_id: v for v in analyzer._assemble(raw)}
    assert out["R01"].confidence == 1.0
    assert out["R04"].confidence == 0.0


def test_min_confidence_filter(rules):
    """Нарушения ниже порога уверенности отсекаются."""
    analyzer = Analyzer(FakeClient(), rules, min_confidence=0.5)
    raw = [
        {"rule_id": "R01", "timestamp": 1.0, "quote": "a",
         "reasoning": "b", "confidence": 0.3},   # ниже порога
        {"rule_id": "R04", "timestamp": 2.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},   # выше
    ]
    out = analyzer._assemble(raw)
    assert [v.rule_id for v in out] == ["R04"]


# ---------------------------------------------------------------------------
# Дедупликация
# ---------------------------------------------------------------------------

def test_dedupe_same_rule_close_timestamp(rules):
    """Одно правило + близкий таймкод (±2с) → один, с большей уверенностью."""
    analyzer = Analyzer(FakeClient(), rules)
    raw = [
        {"rule_id": "R01", "timestamp": 10.0, "quote": "a",
         "reasoning": "b", "confidence": 0.6},
        {"rule_id": "R01", "timestamp": 11.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},   # дубль (разница 1с)
    ]
    out = analyzer._assemble(raw)
    assert len(out) == 1
    assert out[0].confidence == 0.9   # оставлен более уверенный


def test_dedupe_keeps_far_timestamps(rules):
    """То же правило, но далеко по времени (>2с) → два разных нарушения."""
    analyzer = Analyzer(FakeClient(), rules)
    raw = [
        {"rule_id": "R01", "timestamp": 10.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},
        {"rule_id": "R01", "timestamp": 30.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},
    ]
    out = analyzer._assemble(raw)
    assert len(out) == 2


def test_different_rules_not_deduped(rules):
    """Разные правила на одном таймкоде — оба остаются."""
    analyzer = Analyzer(FakeClient(), rules)
    raw = [
        {"rule_id": "R01", "timestamp": 5.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},
        {"rule_id": "R04", "timestamp": 5.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},
    ]
    out = analyzer._assemble(raw)
    assert len(out) == 2


def test_output_sorted_by_timestamp(rules):
    """Итог отсортирован по времени."""
    analyzer = Analyzer(FakeClient(), rules)
    raw = [
        {"rule_id": "R04", "timestamp": 30.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},
        {"rule_id": "R01", "timestamp": 5.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},
    ]
    out = analyzer._assemble(raw)
    assert [v.timestamp for v in out] == [5.0, 30.0]


# ---------------------------------------------------------------------------
# Чанкинг
# ---------------------------------------------------------------------------

def test_chunk_short_protocol_single(rules):
    analyzer = Analyzer(FakeClient(), rules, chunk_lines=60)
    protocol = "\n".join(f"[00:0{i}] x: реплика" for i in range(5))
    chunks = analyzer._chunk(protocol)
    assert len(chunks) == 1


def test_chunk_long_protocol_with_overlap(rules):
    """Длинный протокол режется с перекрытием; последняя строка покрыта."""
    analyzer = Analyzer(FakeClient(), rules, chunk_lines=10, overlap_lines=3)
    lines = [f"line{i}" for i in range(25)]
    chunks = analyzer._chunk("\n".join(lines))

    assert len(chunks) > 1
    # overlap: первая строка второго чанка должна встречаться в первом
    first_lines = chunks[0].splitlines()
    second_lines = chunks[1].splitlines()
    assert second_lines[0] in first_lines
    # последняя строка протокола попала в последний чанк
    assert "line24" in chunks[-1]


def test_chunk_empty(rules):
    analyzer = Analyzer(FakeClient(), rules)
    assert analyzer._chunk("") == []


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------

def test_summary_no_violations(rules):
    analyzer = Analyzer(FakeClient(), rules)
    assert "не выявлено" in analyzer._build_summary([])


def test_summary_counts_by_severity(rules):
    analyzer = Analyzer(FakeClient(), rules)
    raw = [
        {"rule_id": "R04", "timestamp": 1.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},   # critical
        {"rule_id": "R01", "timestamp": 5.0, "quote": "a",
         "reasoning": "b", "confidence": 0.9},   # high
    ]
    summary = analyzer._build_summary(analyzer._assemble(raw))
    assert "2" in summary
    assert "критических" in summary
    assert "высоких" in summary


# ---------------------------------------------------------------------------
# Полный прогон analyze() с фейковым клиентом
# ---------------------------------------------------------------------------

def test_full_analyze_flow(rules, diar):
    """Сквозной прогон: роли + нарушения, severity подставлен, роли применены."""
    client = FakeClient(
        roles_response={"roles": [
            {"speaker": "SPEAKER_00", "role": "следователь"},
            {"speaker": "SPEAKER_01", "role": "обыскиваемый"},
        ]},
        violations_responses=[{
            "violations": [{
                "rule_id": "R04",
                "timestamp": 5.5,
                "quote": "Начинаем осмотр",
                "reasoning": "поиск начат без предложения добровольной выдачи",
                "confidence": 0.85,
            }]
        }],
    )
    analyzer = Analyzer(client, rules)
    result = analyzer.analyze("fake.mp4", diar)

    assert result.speaker_roles["SPEAKER_00"] == "следователь"
    assert len(result.violations) == 1
    v = result.violations[0]
    assert v.rule_id == "R04"
    assert v.severity == Severity.CRITICAL   # из правила
    assert v.confidence == 0.85
    assert "не выявлено" not in result.summary


def test_analyze_survives_role_failure(rules, diar):
    """Если проход ролей упал — анализ продолжается без ролей."""
    class RoleFailClient(FakeClient):
        def generate_json(self, system, prompt, schema):
            from src.llm.client import OllamaError
            if "роль" in system.lower():
                raise OllamaError("симуляция сбоя ролей")
            return {"violations": []}

    analyzer = Analyzer(RoleFailClient(), rules)
    result = analyzer.analyze("fake.mp4", diar)
    assert result.speaker_roles == {}        # роли пустые, но не упали
    assert result.violations == []