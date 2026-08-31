"""FastAPI app for Full Court (NBA + WNBA).

This module only assembles the app. Endpoints live in `routers/`, grouped by
the product surface they serve; shared maths lives in `analytics.py`; and
everything that differs between leagues lives in `leagues.py`.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import (
    ask as ask_router,
    compare as compare_router,
    explorer as explorer_router,
    meta as meta_router,
    players as players_router,
    predictions as predictions_router,
    shots as shots_router,
    similarity as similarity_router,
    teams as teams_router,
)

app = FastAPI(
    title="Full Court API",
    description="Player-season and team analytics for the NBA and WNBA, served from local Parquet.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
def _missing_data(request: Request, exc: FileNotFoundError):
    """A league with no Parquet on disk is an expected, actionable state —
    report it as 404 with the ETL command rather than a 500."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


for router in (
    meta_router.router,
    ask_router.router,
    players_router.router,
    predictions_router.router,
    compare_router.router,
    similarity_router.router,
    shots_router.router,
    teams_router.router,
    explorer_router.router,
):
    app.include_router(router)
