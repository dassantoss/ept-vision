#!/usr/bin/env python3
"""
API router configuration module.

This module configures the main API router and includes all endpoint routers
with their respective prefixes and tags.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    health,
    images,
    users,
    analysis,
    dataset,
    prelabel,
    frontend
)


api_router = APIRouter()


api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(images.router, prefix="/images", tags=["images"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["analysis"]
)
api_router.include_router(dataset.router, prefix="/dataset", tags=["dataset"])
api_router.include_router(prelabel.router, tags=["prelabel"])
api_router.include_router(frontend.router, tags=["frontend"])