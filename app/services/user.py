#!/usr/bin/env python3
"""User service for database operations.

Provides functionality for user management and authentication.
"""

from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.schemas.base import UserCreate


class UserService:
    """Service for managing user operations.

    Provides methods for user creation, retrieval, and authentication.
    """

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email address.

        Args:
            db: Database session
            email: User's email address

        Returns:
            User if found, None otherwise
        """
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def create(db: Session, user_in: UserCreate) -> User:
        """Create a new user.

        Args:
            db: Database session
            user_in: User creation data

        Returns:
            Created user instance
        """
        user = User(
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            full_name=user_in.full_name,
            is_superuser=user_in.is_superuser,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def authenticate(db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate a user.

        Args:
            db: Database session
            email: User's email address
            password: User's password

        Returns:
            User if authentication successful, None otherwise
        """
        user = UserService.get_by_email(db=db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
