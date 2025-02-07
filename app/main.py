#!/usr/bin/env python3
"""Main application module.

This module initializes and configures the FastAPI application, including:
- API routes
- CORS middleware
- Static files
- Templates
- Dataset manager
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os

from app.core.config import settings
from app.api.v1.api import api_router
from app.api.v1.endpoints import frontend
from app.core.logging import get_logger
from app.tools.dataset_manager import DatasetManager


logger = get_logger("ept_vision.main")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


# Configure templates and static files
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Configure CORS
if settings.BACKEND_CORS_ORIGINS:
    origins = [str(origin) for origin in settings.BACKEND_CORS_ORIGINS]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Initialize dataset manager
app.state.dataset_manager = DatasetManager()


# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(frontend.router, tags=["frontend"])


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main labeling interface.

    Args:
        request: FastAPI request object

    Returns:
        HTML response with rendered template

    Raises:
        Exception: If template rendering fails
    """
    try:
        return templates.TemplateResponse(
            "labeler.html",
            {"request": request}
        )
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        raise
