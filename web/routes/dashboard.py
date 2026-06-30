from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from web.auth import get_current_user
from web.context import resolve_lang, resolve_theme, translate_department, user_stats
from web.database import get_db
from web.i18n import get_all, LANG_LABELS
from web.models import Case, User

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
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
        .limit(10)
        .all()
    )

    stats = user_stats(db, user)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "tr": tr,
        "lang": lang,
        "theme": theme,
        "user": user,
        "user_department": translate_department(lang, user.department),
        "cases": cases,
        "stats": stats,
        "active_page": "dashboard",
        "lang_labels": LANG_LABELS,
    })