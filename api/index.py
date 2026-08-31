"""Vercel serverless entrypoint.

Vercel's Python runtime serves the module-level ASGI `app` it finds here. The
application itself lives in `backend/` — this only puts the repo root on the
path, since the function is invoked from `api/` one level below it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app  # noqa: E402

__all__ = ["app"]
