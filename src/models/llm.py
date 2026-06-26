"""Data models for the LLM analysis layer."""

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Числовой ранг для сортировки (critical выше всех)."""
        return {"critical": 3, "high": 2, "medium": 1, "low": 0}[self.value]


@dataclass
class Rule:
    """Одно проверяемое требование из search_procedures.txt."""
    rule_id: str            # "R04"
    severity: Severity
    check: str              # "audio" | "video" | "video+audio"
    text: str               # формулировка требования
    stage: str = ""         # стадия (из заголовка секции)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "check": self.check,
            "text": self.text,
            "stage": self.stage,
        }


@dataclass
class Violation:
    """
    Финальное нарушение в отчёте.
    severity берётся ИЗ ПРАВИЛА (код подставляет), не из ответа модели.
    """
    rule_id: str
    rule_text: str
    severity: Severity
    timestamp: float        # когда (для проверки человеком)
    quote: str              # цитата-доказательство из транскрипции
    reasoning: str          # почему модель сочла это нарушением
    confidence: float       # уверенность модели 0..1
    stage: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "severity": self.severity.value,
            "timestamp": round(self.timestamp, 2),
            "quote": self.quote,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 4),
            "stage": self.stage,
        }


@dataclass
class AnalysisResult:
    """Полный результат LLM-анализа одного дела."""
    file_path: str
    speaker_roles: dict = field(default_factory=dict)   # SPEAKER_00 → "следователь"
    violations: list[Violation] = field(default_factory=list)
    summary: str = ""
    model_used: str = ""

    def violations_by_severity(self) -> list[Violation]:
        return sorted(self.violations, key=lambda v: v.severity.rank, reverse=True)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "speaker_roles": self.speaker_roles,
            "summary": self.summary,
            "model_used": self.model_used,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations_by_severity()],
        }