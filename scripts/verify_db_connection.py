#!/usr/bin/env python3
"""Database connection verification script.

This script verifies and initializes the database connection:
- Checks PostgreSQL server connectivity
- Creates database if it doesn't exist
- Verifies table structure
- Tests connection with basic query

The script ensures that:
- Database server is accessible
- Required database exists
- Schema is properly initialized
- Connection parameters are valid
"""

import os
from typing import List, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.core.logging import get_logger

logger = get_logger("verify_db_connection")


def create_postgres_engine() -> Engine:
    """Create SQLAlchemy engine for PostgreSQL server connection.

    Returns:
        SQLAlchemy Engine instance

    Raises:
        Exception: If connection parameters are invalid
    """
    postgres_url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_SERVER')}:"
        f"{os.getenv('POSTGRES_PORT')}/postgres"
    )
    return create_engine(postgres_url, isolation_level="AUTOCOMMIT")


def create_database_engine() -> Engine:
    """Create SQLAlchemy engine for application database connection.

    Returns:
        SQLAlchemy Engine instance

    Raises:
        Exception: If connection parameters are invalid
    """
    db_url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_SERVER')}:"
        f"{os.getenv('POSTGRES_PORT')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(db_url)


def ensure_database_exists(engine: Engine) -> None:
    """Ensure application database exists, create if needed.

    Args:
        engine: SQLAlchemy Engine for postgres connection

    Raises:
        Exception: If database creation fails
    """
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
            {"db_name": os.getenv('POSTGRES_DB')}
        )
        if not result.fetchone():
            connection.execute(
                text(f"CREATE DATABASE {os.getenv('POSTGRES_DB')}")
            )
            logger.info(
                f"✅ Database {os.getenv('POSTGRES_DB')} created successfully"
            )
        else:
            logger.info(
                f"✅ Database {os.getenv('POSTGRES_DB')} already exists"
            )


def get_existing_tables(engine: Engine) -> List[str]:
    """Get list of existing tables in the database.

    Args:
        engine: SQLAlchemy Engine for database connection

    Returns:
        List of table names

    Raises:
        Exception: If query fails
    """
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """))
        return [row[0] for row in result]


def verify_database_connection() -> None:
    """Verify database connectivity and initialization.

    Performs complete database verification:
    1. Connects to PostgreSQL server
    2. Creates database if needed
    3. Tests database connection
    4. Lists existing tables

    Raises:
        Exception: If any verification step fails
    """
    try:
        logger.info("Verifying database connection...")

        # Connect to postgres and create database if needed
        postgres_engine = create_postgres_engine()
        ensure_database_exists(postgres_engine)

        # Connect to application database
        db_engine = create_database_engine()

        with db_engine.connect() as connection:
            # Test connection
            connection.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")

            # List existing tables
            tables = get_existing_tables(db_engine)
            logger.info(f"📋 Found tables: {tables}")

    except Exception as e:
        logger.error(f"❌ Database connection error: {str(e)}")
        raise


if __name__ == "__main__":
    verify_database_connection()
