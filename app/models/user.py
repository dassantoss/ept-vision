#!/usr/bin/env python3
"""User model for database operations.

Defines the User model with SQLAlchemy for database interactions.
"""

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from app.core.database import Base


class User(Base):
    """User model for authentication and authorization.

    Attributes:
        id: Unique identifier
        email: User's email address
        full_name: User's full name
        hashed_password: Encrypted password
        is_active: Account status
        is_superuser: Admin privileges flag
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now()
    )
