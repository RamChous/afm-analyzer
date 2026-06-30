import shutil
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from web.auth import get_current_user
from web.context import resolve_lang, resolve_theme, user_stats
from web.database import get_db
from web.i18n import get_all, DEFAULT_LANG, SUPPORTED_LANGS, LANG_LABELS
from web.jobs import submit_analysis
from web.models import Analysis, Case, User

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
TOTAL_STEPS = 6

# GET /upload — show upload form

@router.get("/upload", response_class=HTMLResponse)
def upload_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lang = resolve_lang(request, user)
    theme = resolve_theme(request, user)
    tr = get_all(lang)
    stats = user_stats(db, user)

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "tr": tr,
        "lang": lang,
        "theme": theme,
        "user": user,
        "stats": stats,
        "active_page": "upload",
        "error": None,
        "lang_labels": LANG_LABELS,
    })

# POST /upload — receive file, create case + analysis rows, queue the job

@router.post("/upload")
async def upload_submit(
    request: Request,
    case_number: str = Form(...),
    search_date: str = Form(""),
    address: str = Form(""),
    lang: str = Form(DEFAULT_LANG),
    video: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG
    theme = resolve_theme(request, user)
    tr = get_all(lang)

    def _error(message: str, status_code: int = 400):
        stats = user_stats(db, user)
        return templates.TemplateResponse("upload.html", {
            "request": request, "tr": tr, "lang": lang, "theme": theme,
            "user": user, "stats": stats, "active_page": "upload",
            "error": message, "lang_labels": LANG_LABELS,
        }, status_code=status_code)

    if not case_number.strip() or not video.filename:
        return _error(tr["upload_error_required"])

    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error(tr["upload_error_format"].format(formats=", ".join(sorted(ALLOWED_EXTENSIONS))))

    safe_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{video.filename}"
    video_path = UPLOAD_DIR / safe_name
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    case = Case(
        case_number=case_number.strip(),
        search_date=search_date.strip() or None,
        address=address.strip() or None,
        owner_id=user.id,
    )
    db.add(case)
    db.flush()

    analysis = Analysis(
        case_id=case.id,
        video_filename=video.filename,
        video_path=str(video_path),
        status="pending",
        current_step=0,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    submit_analysis(analysis.id)

    return RedirectResponse(url=f"/status/{analysis.id}", status_code=302)

# GET /status/{analysis_id} — processing progress page

@router.get("/status/{analysis_id}", response_class=HTMLResponse)
def status_page(
    analysis_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lang = resolve_lang(request, user)
    theme = resolve_theme(request, user)
    tr = get_all(lang)

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return RedirectResponse(url="/dashboard")

    case = db.query(Case).filter(
        Case.id == analysis.case_id, Case.owner_id == user.id,
    ).first()
    if not case:
        return RedirectResponse(url="/dashboard")

    if analysis.status == "cancelled":
        return RedirectResponse("/dashboard")
    if analysis.status == "done":
        return RedirectResponse(f"/cases/{case.id}")

    stats = user_stats(db, user)

    return templates.TemplateResponse("processing.html", {
        "request": request,
        "tr": tr,
        "lang": lang,
        "theme": theme,
        "user": user,
        "analysis": analysis,
        "case": case,
        "stats": stats,
        "total_steps": TOTAL_STEPS,
        "active_page": "cases",
        "lang_labels": LANG_LABELS,
    })

# GET /api/status/{analysis_id} — JSON status for polling

@router.get("/api/status/{analysis_id}")
def api_status(
    analysis_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return JSONResponse({"error": "not found"}, status_code=404)

    case = db.query(Case).filter(
        Case.id == analysis.case_id, Case.owner_id == user.id,
    ).first()
    if not case:
        return JSONResponse({"error": "forbidden"}, status_code=403)

    return JSONResponse({
        "status": analysis.status,
        "current_step": analysis.current_step,
        "step_label": analysis.step_label,
        "total_steps": TOTAL_STEPS,
        "done": analysis.status == "done",
        "cancelled": analysis.status == "cancelled",
        "error": analysis.error_message,
        "redirect": f"/cases/{analysis.case_id}" if analysis.status == "done" else None,
    })

# POST /cases/{analysis_id}/cancel — cancel a pending/processing analysis

@router.post("/analysis/{analysis_id}/cancel")
def cancel_analysis(
    analysis_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    analysis = (
        db.query(Analysis)
        .join(Case)
        .filter(
            Analysis.id == analysis_id,
            Case.owner_id == user.id
        )
        .first()
    )

    if not analysis:
        raise HTTPException(404)

    if analysis.status in ("done", "error", "cancelled"):
        return JSONResponse({"success": False})

    analysis.status = "cancelled"
    db.commit()
    return RedirectResponse(
        url=f"/dashboard&{user.lang}",
        status_code=303
        )