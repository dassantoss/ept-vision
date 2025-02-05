#!/usr/bin/env python3
"""
Health check endpoints for the API.

This module provides endpoints to check the health status of all services
including database and cache connections.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.core.cache import cache
from app.core.logging import get_logger


logger = get_logger("ept_vision.health")
router = APIRouter()


@router.get("/health")
def health_check(
    db: Session = Depends(get_db)
) -> dict:
    """
    Check the health of all services.

    Args:
        db: Database session

    Returns:
        dict: Health status of all services including database and cache

    Raises:
        None: Exceptions are caught and reported in the response
    """
    logger.info("Starting health check...")

    try:
        logger.info("Checking database connection...")
        db.execute(text("SELECT 1"))
        db_status = "healthy"
        logger.info("Database check passed")
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "unhealthy"

    try:
        logger.info("Checking Redis connection...")
        cache.redis_client.ping()
        cache_status = "healthy"
        logger.info("Redis check passed")
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        cache_status = "unhealthy"

    # Check if all services are healthy
    services_status = [db_status, cache_status]
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status
    ) else "unhealthy"
    logger.info(f"Health check completed. Overall status: {overall_status}")

    response = {
        "status": overall_status,
        "version": "1.0.0",
        "services": {
            "database": db_status,
            "cache": cache_status
        }
    }
    logger.info(f"Returning response: {response}")
    return response
