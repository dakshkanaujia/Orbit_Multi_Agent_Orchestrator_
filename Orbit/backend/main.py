"""
main.py — fixes:
H9: API key authentication via X-API-Key header
M8: Upload size middleware (413 before file reaches router)
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# override=False means container env vars (set by docker-compose) always win over the .env file
load_dotenv(override=False)

from fastapi import FastAPI, Request, Depends, Security, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from memory.db import create_tables, close_pool
from routers import captures, items, actions, search, dashboard, hub

# ── API Key Auth (H9) ──────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
ORBIT_API_KEY = os.getenv("ORBIT_API_KEY", "")


def verify_api_key(api_key: str = Security(_API_KEY_HEADER)):
    """H9: reject requests without the correct API key. Skip if ORBIT_API_KEY not set (dev mode)."""
    if not ORBIT_API_KEY:
        return  # dev mode — no key configured, allow all
    if api_key != ORBIT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield
    await close_pool()


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Orbit AI",
    description="Multi-agent personal chief-of-staff: capture → extract → act",
    version="1.0.0",
    lifespan=lifespan,
)

# M8: upload size middleware — reject oversized requests before they reach routers
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"Request body exceeds maximum allowed size ({MAX_UPLOAD_BYTES} bytes)."},
        )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Protected routers (H9)
_auth_dep = [Depends(verify_api_key)]

app.include_router(captures.router,  dependencies=_auth_dep)
app.include_router(items.router,     dependencies=_auth_dep)
app.include_router(actions.router,   dependencies=_auth_dep)
app.include_router(search.router,    dependencies=_auth_dep)
app.include_router(dashboard.router, dependencies=_auth_dep)
app.include_router(hub.router,       dependencies=_auth_dep)


@app.get("/health")
async def health():
    return {"status": "ok"}
