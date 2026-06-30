import io
from pathlib import Path

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from web.auth import get_current_user
from web.context import resolve_lang, resolve_theme, user_stats
from web.database import get_db
from web.i18n import get_all, LANG_LABELS
from web.models import User, Case, Analysis

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")


@router.get("/cases", response_class=HTMLResponse)
def cases_list(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lang = resolve_lang(request, user)
    theme = resolve_theme(request, user)
    tr = get_all(lang)

    cases = (
        db.query(Case)
        .filter(Case.owner_id == user.id)
        .order_by(Case.created_at.desc())
        .all()
    )
    stats = user_stats(db, user)

    return templates.TemplateResponse("cases.html", {
        "request": request,
        "tr": tr,
        "lang": lang,
        "theme": theme,
        "user": user,
        "cases": cases,
        "stats": stats,
        "active_page": "cases",
        "lang_labels": LANG_LABELS,
    })


@router.get("/cases/{case_id}", response_class=HTMLResponse)
def case_results(
    case_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lang = resolve_lang(request, user)
    theme = resolve_theme(request, user)
    tr = get_all(lang)

    case = db.query(Case).filter(
        Case.id == case_id, Case.owner_id == user.id,
    ).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    analysis = case.analysis
    if analysis and analysis.status != "done":
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/status/{analysis.id}")

    stats = user_stats(db, user)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "tr": tr,
        "lang": lang,
        "theme": theme,
        "user": user,
        "case": case,
        "analysis": analysis,
        "result": (analysis.result_json or {}) if analysis else {},
        "stats": stats,
        "active_page": "cases",
        "lang_labels": LANG_LABELS,
    })


@router.get("/cases/{case_id}/export/json")
def export_json_route(
    case_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = db.query(Case).filter(
        Case.id == case_id, Case.owner_id == user.id,
    ).first()
    if not case or not case.analysis or case.analysis.status != "done":
        raise HTTPException(status_code=404)

    payload = {
        "case_number": case.case_number,
        "search_date": case.search_date,
        "address": case.address,
        "analysis": case.analysis.result_json or {},
    }
    return JSONResponse(content=payload, headers={
        "Content-Disposition": f'attachment; filename="case_{case.case_number}.json"'
    })


@router.get("/cases/{case_id}/export/excel")
def export_excel_route(
    case_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Rebuilds an AnalysisResult-shaped object from the stored result_json
    and reuses the existing src/reports/excel_export.py exporter — no
    duplicate report-building logic, no changes to src/.
    """
    case = db.query(Case).filter(
        Case.id == case_id, Case.owner_id == user.id,
    ).first()
    if not case or not case.analysis or case.analysis.status != "done":
        raise HTTPException(status_code=404)

    from src.models.llm import AnalysisResult, Violation, Severity
    from src.reports.excel_export import export_excel

    data = case.analysis.result_json or {}
    violations = [
        Violation(
            rule_id=v["rule_id"],
            rule_text=v["rule_text"],
            severity=Severity(v["severity"]),
            timestamp=v["timestamp"],
            quote=v.get("quote", ""),
            reasoning=v.get("reasoning", ""),
            confidence=v.get("confidence", 0.0),
            stage=v.get("stage", ""),
        )
        for v in data.get("violations", [])
    ]
    result = AnalysisResult(
        file_path=case.analysis.video_path or "",
        speaker_roles=data.get("speaker_roles", {}),
        violations=violations,
        summary=case.analysis.llm_summary or "",
        model_used=case.analysis.model_used or "",
    )

    tmp_path = Path(f"/tmp/case_{case.id}_report.xlsx")
    export_excel(result, str(tmp_path))

    file_bytes = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="case_{case.case_number}_report.xlsx"'
        },
    )