"""
src/llm/analyzer.py
Оркестратор LLM-анализа. Связывает диаризацию + timeline + правила
в единый AnalysisResult через четыре прохода:

    0. build_protocol      — текст с таймкодами (без LLM)
    1. identify_roles      — роли по началу записи (1 вызов LLM)
    2. find_violations     — поиск нарушений по чанкам (map)
    3. assemble            — дедуп + подстановка severity из правил (reduce)

Сам ничего тяжёлого не запускает — потребляет готовые результаты.
"""

import logging
from typing import Optional

from src.models.speech import DiarizationResult
from src.models.video import Timeline
from src.models.llm import Rule, Severity, Violation, AnalysisResult
from src.llm.client import LLMClient, OllamaError
from src.llm.rules import rules_index
from src.llm.schemas import SPEAKER_ROLES_SCHEMA, VIOLATIONS_SCHEMA
from src.llm import prompts

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Usage:
        from src.llm.client import LLMClient
        from src.llm.rules import parse_rules

        client = LLMClient(model="qwen3:30b", num_ctx=8192)
        rules = parse_rules("config/procedures/search_procedures.txt")

        analyzer = Analyzer(client, rules)
        result = analyzer.analyze(
            file_path="video.mp4",
            diarization=diarization_result,
            timeline=timeline_result,   # опционально
        )
    """

    def __init__(
        self,
        client: LLMClient,
        rules: list[Rule],
        roles_max_utterances: int = 40,
        roles_max_seconds: float = 300.0,
        chunk_lines: int = 60,
        overlap_lines: int = 10,
        min_confidence: float = 0.0,
    ):
        """
        Args:
            client:               LLM-клиент (Ollama).
            rules:                распарсенные правила.
            roles_max_utterances: сколько реплик от начала брать для ролей.
            roles_max_seconds:    ...или сколько секунд (что раньше).
            chunk_lines:          размер чанка протокола в строках.
            overlap_lines:        перекрытие между чанками (чтобы не резать
                                  нарушение на границе).
            min_confidence:       нарушения с confidence ниже порога отбрасываются.
        """
        self.client = client
        self.rules = rules
        self._rules_idx = rules_index(rules)
        self.roles_max_utterances = roles_max_utterances
        self.roles_max_seconds = roles_max_seconds
        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        file_path: str,
        diarization: DiarizationResult,
        timeline: Optional[Timeline] = None,
    ) -> AnalysisResult:
        logger.info("LLM-анализ: %s (%d реплик, %d правил)",
                    file_path, len(diarization.utterances), len(self.rules))

        # Проход 1: роли по началу записи
        roles = self._identify_roles(diarization)
        logger.info("Роли определены: %s", roles or "—")

        # Проход 0+2: полный протокол с подставленными ролями → чанки → нарушения
        protocol = prompts.build_protocol(diarization, timeline, roles)
        violations = self._find_violations(protocol, roles)

        result = AnalysisResult(
            file_path=file_path,
            speaker_roles=roles,
            violations=violations,
            summary=self._build_summary(violations),
            model_used=self.client.model,
        )
        logger.info("Анализ завершён: %d нарушений", len(violations))
        return result

    # ------------------------------------------------------------------
    # Проход 1: роли
    # ------------------------------------------------------------------

    def _identify_roles(self, diarization: DiarizationResult) -> dict[str, str]:
        """Определяет роли по началу записи. При сбое — пустой маппинг (не падаем)."""
        head = self._protocol_head(diarization)
        if not head:
            return {}

        # Лёгкий under-protocol только из головы, без timeline
        head_diar = DiarizationResult(
            file_path=diarization.file_path,
            num_speakers=diarization.num_speakers,
            utterances=head,
        )
        protocol = prompts.build_protocol(head_diar)

        try:
            data = self.client.generate_json(
                system=prompts.SPEAKER_ROLES_SYSTEM,
                prompt=prompts.build_roles_prompt(protocol),
                schema=SPEAKER_ROLES_SCHEMA,
            )
        except OllamaError as exc:
            logger.warning("Не удалось определить роли (%s) — продолжаю без них", exc)
            return {}

        roles: dict[str, str] = {}
        for item in data.get("roles", []):
            speaker = item.get("speaker")
            role = item.get("role")
            if speaker and role:
                roles[speaker] = role
        return roles

    def _protocol_head(self, diarization: DiarizationResult):
        """Начало записи для ролей: ограничено и по репликам, и по времени."""
        utts = sorted(diarization.utterances, key=lambda u: u.start)
        head = []
        for u in utts:
            if len(head) >= self.roles_max_utterances:
                break
            if u.start > self.roles_max_seconds:
                break
            head.append(u)
        return head

    # ------------------------------------------------------------------
    # Проход 2: нарушения по чанкам
    # ------------------------------------------------------------------

    def _find_violations(
        self,
        protocol: str,
        roles: dict[str, str],
    ) -> list[Violation]:
        chunks = self._chunk(protocol)
        logger.info("Протокол разбит на %d чанков", len(chunks))

        raw: list[dict] = []
        for i, chunk in enumerate(chunks, 1):
            try:
                data = self.client.generate_json(
                    system=prompts.VIOLATIONS_SYSTEM,
                    prompt=prompts.build_violations_prompt(chunk, self.rules, roles),
                    schema=VIOLATIONS_SCHEMA,
                )
            except OllamaError as exc:
                logger.warning("Чанк %d/%d: сбой LLM (%s) — пропускаю",
                               i, len(chunks), exc)
                continue
            raw.extend(data.get("violations", []))

        return self._assemble(raw)

    def _chunk(self, protocol: str) -> list[str]:
        """Режет протокол на перекрывающиеся куски по строкам."""
        lines = protocol.splitlines()
        if not lines:
            return []
        if len(lines) <= self.chunk_lines:
            return [protocol]

        step = max(1, self.chunk_lines - self.overlap_lines)
        chunks = []
        start = 0
        while start < len(lines):
            chunk = lines[start:start + self.chunk_lines]
            chunks.append("\n".join(chunk))
            if start + self.chunk_lines >= len(lines):
                break
            start += step
        return chunks

    # ------------------------------------------------------------------
    # Проход 3: сборка (валидация + severity + дедуп)
    # ------------------------------------------------------------------

    def _assemble(self, raw: list[dict]) -> list[Violation]:
        """
        Превращает сырые ответы модели в Violation:
        - валидирует rule_id (отбрасывает несуществующие — защита от галлюцинаций);
        - подставляет severity и текст ИЗ ПРАВИЛА, а не из ответа модели;
        - фильтрует по min_confidence;
        - дедуплицирует (overlap чанков даёт дубли).
        """
        violations: list[Violation] = []

        for item in raw:
            rule_id = item.get("rule_id", "").strip()
            rule = self._rules_idx.get(rule_id)
            if rule is None:
                logger.warning("Модель сослалась на неизвестное правило '%s' — отброшено",
                               rule_id)
                continue

            confidence = self._clamp(item.get("confidence", 0.0))
            if confidence < self.min_confidence:
                continue

            violations.append(Violation(
                rule_id=rule.rule_id,
                rule_text=rule.text,          # из правила, не из модели
                severity=rule.severity,       # из правила, не из модели
                stage=rule.stage,
                timestamp=float(item.get("timestamp", 0.0)),
                quote=item.get("quote", "").strip(),
                reasoning=item.get("reasoning", "").strip(),
                confidence=confidence,
            ))

        return self._dedupe(violations)

    def _dedupe(self, violations: list[Violation]) -> list[Violation]:
        """
        Схлопывает дубли по (rule_id, таймкод ±2с). Оставляет вариант
        с наибольшей уверенностью. Overlap между чанками — главный источник дублей.
        """
        kept: list[Violation] = []
        for v in sorted(violations, key=lambda x: x.confidence, reverse=True):
            is_dup = any(
                v.rule_id == k.rule_id and abs(v.timestamp - k.timestamp) <= 2.0
                for k in kept
            )
            if not is_dup:
                kept.append(v)
        return sorted(kept, key=lambda v: v.timestamp)

    @staticmethod
    def _clamp(value, lo: float = 0.0, hi: float = 1.0) -> float:
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Сводка (детерминированная — без LLM, чтобы не плодить фантазии)
    # ------------------------------------------------------------------

    def _build_summary(self, violations: list[Violation]) -> str:
        if not violations:
            return "Нарушений по проверяемым требованиям не выявлено."

        by_sev: dict[Severity, int] = {}
        for v in violations:
            by_sev[v.severity] = by_sev.get(v.severity, 0) + 1

        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        names = {Severity.CRITICAL: "критических", Severity.HIGH: "высоких",
                 Severity.MEDIUM: "средних", Severity.LOW: "низких"}
        parts = [f"{by_sev[s]} {names[s]}" for s in order if s in by_sev]
        return f"Выявлено нарушений: {len(violations)} ({', '.join(parts)})."


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_search(
    file_path: str,
    diarization: DiarizationResult,
    rules: list[Rule],
    model: str,
    timeline: Optional[Timeline] = None,
    num_ctx: int = 8192,
) -> AnalysisResult:
    """One-liner для analysis_pipeline.py."""
    client = LLMClient(model=model, num_ctx=num_ctx)
    return Analyzer(client, rules).analyze(
        file_path=file_path,
        diarization=diarization,
        timeline=timeline,
    )