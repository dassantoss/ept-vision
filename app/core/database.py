#!/usr/bin/env python3
"""
Database configuration module.

This module provides database connection management and session handling
using SQLAlchemy, including connection URL construction and session
lifecycle management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator

from app.core.config import settings
from app.core.logging import get_logger


logger = get_logger("database")

# Construct database URL from settings
SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:"
    f"{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:"
    f"{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

# Create database engine with connection pool
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True  # Enable connection health checks
)

# Configure session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base for models
Base = declarative_base()


def get_db() -> Generator:
    """
    Get database session from connection pool.

    This function creates a new database session and ensures proper
    cleanup after use, even if exceptions occur.

    Yields:
        Session: Database session object

    Raises:
        Exception: Any exception that occurs during session use is
                  propagated after ensuring connection cleanup
    """
    db = SessionLocal()
    logger.debug("Creating new database session")
    try:
        yield db
    finally:
        logger.debug("Closing database session")
        db.close()
