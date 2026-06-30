from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from web.auth import authenticate_user, delete_session, login_user
from web.context import resolve_lang, resolve_theme
from web.database import get_db
from web.i18n import DEFAULT_LANG, LANG_LABELS, SUPPORTED_LANGS, get_all

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

ONE_YEAR = 60 * 60 * 24 * 365
SESSION_MAX_AGE = 60 * 60 * 8  # 8h shift


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    lang = resolve_lang(request)
    theme = resolve_theme(request)
    response = templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "tr": get_all(lang),
            "lang": lang,
            "theme": theme,
            "lang_labels": LANG_LABELS,
            "error": None,
            "username": "",
        },
    )
    response.set_cookie("lang", lang, max_age=ONE_YEAR)
    response.set_cookie("theme", theme, max_age=ONE_YEAR)
    return response


@router.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    lang: str = Form(DEFAULT_LANG),
    db: Session = Depends(get_db),
):
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG
    theme = resolve_theme(request)

    tr = get_all(lang)
    username = username.strip()

    if not username or not password:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request, "tr": tr, "lang": lang, "theme": theme,
                "lang_labels": LANG_LABELS,
                "error": tr["login_error_empty"],
                "username": username,
            },
            status_code=400,
        )

    user = authenticate_user(username, password, db)

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request, "tr": tr, "lang": lang, "theme": theme,
                "lang_labels": LANG_LABELS,
                "error": tr["login.error.invalid"],
                "username": username,
            },
            status_code=401,
        )

    if not user.is_active:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request, "tr": tr, "lang": lang, "theme": theme,
                "lang_labels": LANG_LABELS,
                "error": tr["login.error.inactive"],
                "username": username,
            },
            status_code=403,
        )

    token = login_user(user, db)
    user.lang = lang
    db.commit()

    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key="session", value=token,
        httponly=True, samesite="lax", max_age=SESSION_MAX_AGE,
    )
    response.set_cookie("lang", lang, max_age=ONE_YEAR)
    response.set_cookie("theme", theme, max_age=ONE_YEAR)
    return response


@router.get("/logout")
def logout(request: Request, response: Response):
    session = request.cookies.get("session")
    if session:
        delete_session(session)

    redirect = RedirectResponse(url="/login", status_code=302)
    redirect.delete_cookie("session")
    return redirect