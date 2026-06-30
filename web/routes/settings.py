"""
GET /settings           -> settings page (theme, language, account info)
POST /settings/theme     -> flip dark/light, persist to cookie + user profile,
                            redirect back to the referring page
POST /settings/lang      -> set language, persist to cookie + user profile
"""

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from web.auth import get_current_user
from web.context import resolve_lang, resolve_theme, SUPPORTED_THEMES
from web.database import get_db
from web.i18n import get_all, SUPPORTED_LANGS, LANG_LABELS
from web.models import User

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

ONE_YEAR = 60 * 60 * 24 * 365


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lang = resolve_lang(request, user)
    theme = resolve_theme(request, user)
    tr = get_all(lang)

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "tr": tr,
        "lang": lang,
        "theme": theme,
        "lang_labels": LANG_LABELS,
        "user": user,
        "active_page": "settings",
    })


@router.post("/settings/theme")
def toggle_theme(
    request: Request,
    redirect_to: str = Form("/dashboard"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current = resolve_theme(request, user)
    new_theme = "light" if current == "dark" else "dark"

    user.theme = new_theme
    db.commit()

    safe_redirect = redirect_to if redirect_to.startswith("/") else "/dashboard"
    response = RedirectResponse(url=safe_redirect, status_code=302)
    response.set_cookie("theme", new_theme, max_age=ONE_YEAR)
    return response


@router.post("/settings/lang")
def set_lang(
    request: Request,
    lang: str = Form(...),
    redirect_to: str = Form("/dashboard"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if lang not in SUPPORTED_LANGS:
        return RedirectResponse(url="/settings", status_code=302)

    user.lang = lang
    db.commit()

    safe_redirect = redirect_to if redirect_to.startswith("/") else "/dashboard"
    response = RedirectResponse(url=safe_redirect, status_code=302)
    response.set_cookie("lang", lang, max_age=ONE_YEAR)
    return response