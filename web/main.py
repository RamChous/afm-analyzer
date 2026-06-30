import logging

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.database import init_db
from web.jobs import shutdown_executor
from web.routes.auth import router as auth_router
from web.routes.dashboard import router as dashboard_router
from web.routes.cases import router as cases_router
from web.routes.analysis import router as analysis_router
from web.routes.settings import router as settings_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="Inspectra",
    description="Система анализа видеозаписей обысков АФМ РК",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")


@app.on_event("startup")
def startup():
    print("Inspectra запускается...")
    init_db()
    print("База данных готова")
    print("Тестовые пользователи: admin/admin123, asel/asel123, daulet/daulet123")


@app.on_event("shutdown")
def shutdown():
    shutdown_executor()


app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(cases_router)
app.include_router(analysis_router)
app.include_router(settings_router)


@app.get("/")
def root():
    return RedirectResponse(url="/login")


@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc):
    return RedirectResponse(url="/login")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return RedirectResponse(url="/dashboard")