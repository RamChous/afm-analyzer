from fastapi import Request
from sqlalchemy.orm import Session

from web.i18n import DEFAULT_LANG, SUPPORTED_LANGS, get_all
from web.models import Analysis, Case, User

SUPPORTED_THEMES = ["dark", "light"]
DEFAULT_THEME = "dark"

DEPARTMENT_KEYS = ["dept_prevention", "dept_financial"]

DEPARTMENT_LEGACY_MAP = {
    "Отдел управления превенции и аналитических разработок": "dept_prevention",
    "Отдел финансовых расследований": "dept_financial",
}


def translate_department(lang: str, department: str | None) -> str:
    if not department:
        return ""
    key = department if department in DEPARTMENT_KEYS else DEPARTMENT_LEGACY_MAP.get(department)
    if key:
        return get_all(lang).get(key, department)
    return department


def resolve_lang(request: Request, user: User | None = None) -> str:
    lang = request.query_params.get("lang")
    if lang not in SUPPORTED_LANGS:
        cookie_lang = request.cookies.get("lang")
        lang = cookie_lang if cookie_lang in SUPPORTED_LANGS else None
    if lang not in SUPPORTED_LANGS and user is not None:
        lang = user.lang if user.lang in SUPPORTED_LANGS else None
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG
    return lang


def resolve_theme(request: Request, user: User | None = None) -> str:
    theme = request.cookies.get("theme")
    if theme not in SUPPORTED_THEMES and user is not None:
        theme = user.theme if user.theme in SUPPORTED_THEMES else None
    if theme not in SUPPORTED_THEMES:
        theme = DEFAULT_THEME
    return theme


def user_stats(db: Session, user: User) -> dict:
    analyses = (
        db.query(Analysis)
        .join(Case)
        .filter(Case.owner_id == user.id)
        .all()
    )
    done = [a for a in analyses if a.status == "done"]
    return {
        "total": len(analyses),
        "violations": len([a for a in done if a.violation_count and a.violation_count > 0]),
        "clean": len([a for a in done if not a.violation_count]),
        "processing": len([a for a in analyses if a.status in ("pending", "processing")]),
        "cases_with_violations": len({
            a.case_id for a in done if a.violation_count and a.violation_count > 0
        }),
    }