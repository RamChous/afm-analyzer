import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from web.database import SessionLocal
from web.models import Analysis

logger = logging.getLogger("inspectra.pipeline")

STEP_VIDEO = 1
STEP_TRANSCRIBE = 2
STEP_DIARIZE = 3
STEP_TIMELINE = 4
STEP_LLM = 5
STEP_SAVE = 6
TOTAL_STEPS = 6

RULES_PATH = os.environ.get(
    "RULES_PATH", "config/procedures/search_procedures.txt"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:14b")
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "8192"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
YOLO_SAMPLE_RATE = float(os.environ.get("YOLO_SAMPLE_RATE", "1.0"))


def _derive_verdict(violations, obstruction=None) -> str:
    severities = {v.severity.value for v in violations}

    if obstruction is not None and obstruction.has_obstructions:
        severities.add("high")

    if not severities:
        return "СООТВЕТСТВУЕТ"
    if "critical" in severities or "high" in severities:
        return "НЕ_СООТВЕТСТВУЕТ"
    return "ЧАСТИЧНО_СООТВЕТСТВУЕТ"


def run_pipeline(analysis_id: int) -> None:
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            logger.error("Analysis %s not found, aborting pipeline", analysis_id)
            return

        def set_step(step: int, label: str, status: str = "processing"):
            analysis.current_step = step
            analysis.step_label = label
            analysis.status = status
            db.commit()
        
        def is_cancelled():
            db.refresh(analysis)
            return analysis.status == "cancelled"

        video_path = analysis.video_path
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            logger.warning("HF_TOKEN is not set — diarization will fail.")

        analysis.started_at = datetime.utcnow()
        db.commit()

        set_step(STEP_VIDEO, "video")
        from src.video.frame_analysis import analyze_frames
        from src.video.person_detection import detect_persons
        from src.video.camera_obstruction import detect_obstructions

        video_result = analyze_frames(video_path, sample_rate=YOLO_SAMPLE_RATE)
        persons = detect_persons(video_result)
        obstruction = detect_obstructions(video_result)

        analysis.video_duration = video_result.duration
        analysis.participant_by_video = persons.max_count
        analysis.camera_obstructed = obstruction.has_obstructions
        analysis.camera_obstruction_sec = obstruction.total_obstructed_seconds
        db.commit()

        if is_cancelled():
            logger.info("Analysis cancelled")
            return

        set_step(STEP_TRANSCRIBE, "transcribe")
        from src.speech.transcription import transcribe_video

        transcription = transcribe_video(
            video_path, model_size=WHISPER_MODEL, language=None
        )
        analysis.language = transcription.language
        if transcription.duration:
            analysis.video_duration = transcription.duration
        analysis.full_transcript = transcription.full_text
        db.commit()

        if is_cancelled():
            logger.info("Analysis cancelled")
            return

        set_step(STEP_DIARIZE, "diarize")
        from src.speech.diarization import diarize_video

        diarization = diarize_video(
            file_path=video_path,
            transcription=transcription,
            hf_token=hf_token,
            speaker_count=persons.max_count or None,
        )
        analysis.participant_by_voice = diarization.num_speakers
        analysis.participant_count = max(
            persons.max_count or 0, diarization.num_speakers or 0
        ) or None
        db.commit()

        if is_cancelled():
            logger.info("Analysis cancelled")
            return

        set_step(STEP_TIMELINE, "timeline")
        from src.video.timeline_events import build_timeline

        timeline = build_timeline(
            video_result=video_result,
            obstruction=obstruction,
            diarization=diarization,
        )

        if is_cancelled():
            logger.info("Analysis cancelled")
            return 
        
        set_step(STEP_LLM, "llm")
        from src.llm.rules import parse_rules
        from src.llm.analyzer import analyze_search

        rules = parse_rules(RULES_PATH)
        result = analyze_search(
            file_path=video_path,
            diarization=diarization,
            rules=rules,
            model=LLM_MODEL,
            timeline=timeline,
            num_ctx=LLM_NUM_CTX,
        )

        if is_cancelled():
            logger.info("Analysis cancelled")
            return

        violations = list(result.violations)

        obstruction_rule = next((r for r in rules if r.rule_id == "R13"), None)
        already_flagged = any(v.rule_id == "R13" for v in violations)
        if obstruction.has_obstructions and obstruction_rule and not already_flagged:
            from src.models.llm import Violation as _Violation

            first_event = obstruction.events[0] if obstruction.events else None
            violations.append(_Violation(
                rule_id=obstruction_rule.rule_id,
                rule_text=obstruction_rule.text,
                severity=obstruction_rule.severity,
                stage=obstruction_rule.stage,
                timestamp=first_event.start if first_event else 0.0,
                quote="",
                reasoning=(
                    "Перекрытие камеры обнаружено автоматическим анализом "
                    f"видео ({obstruction.total_obstructed_seconds:.1f} с суммарно), "
                    "не было явно отражено в текстовой расшифровке."
                ),
                confidence=max((e.confidence for e in obstruction.events), default=0.8),
            ))

        analysis.violation_count = len(violations)
        analysis.llm_summary = result.summary
        analysis.model_used = result.model_used
        analysis.verdict = _derive_verdict(violations, obstruction)

        set_step(STEP_SAVE, "save")

        sorted_violations = sorted(violations, key=lambda v: v.severity.rank, reverse=True)

        analysis.result_json = {
            "speaker_roles": result.speaker_roles,
            "summary": result.summary,
            "verdict": analysis.verdict,
            "violation_count": len(violations),
            "violations": [v.to_dict() for v in sorted_violations],
            "participants": {
                "total_count": analysis.participant_count,
                "by_video": analysis.participant_by_video,
                "by_voice": analysis.participant_by_voice,
            },
            "camera_obstruction": {
                "detected": obstruction.has_obstructions,
                "total_seconds": obstruction.total_obstructed_seconds,
                "events": [e.to_dict() for e in obstruction.events],
            },
            "transcript": {
                "language": transcription.language,
                "full_text": transcription.full_text,
                "utterances": [u.to_dict() for u in diarization.utterances],
            },
            "rules_checked": [r.to_dict() for r in rules],
        }

        if is_cancelled():
            logger.info("Analysis cancelled")
            return

        analysis.status = "done"
        if is_cancelled():
            return
        
        analysis.finished_at = datetime.utcnow()
        db.commit()
        logger.info("Analysis %s done: %d violations, verdict=%s",
                    analysis_id, len(result.violations), analysis.verdict)

    except Exception as exc:
        db.rollback()
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.status = "error"
            analysis.error_message = str(exc)
            analysis.finished_at = datetime.utcnow()
            db.commit()
        logger.exception("Pipeline failed for analysis %s", analysis_id)
    finally:
        db.close()