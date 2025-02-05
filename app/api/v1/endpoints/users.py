#!/usr/bin/env python3
"""
User management endpoints for the API.

This module provides endpoints for user operations including creation
and retrieval of user information.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.schemas.user import User
from app.core.logging import get_logger
from app.schemas.base import UserCreate, UserInDBBase
from app.services.user import UserService

logger = get_logger("ept_vision.users_api")
router = APIRouter()


@router.post("/", response_model=UserInDBBase)
def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db)
) -> Any:
    """
    Create new user.

    Args:
        user_in: User creation data
        db: Database session

    Returns:
        UserInDBBase: Created user data

    Raises:
        HTTPException: If user with same email already exists
    """
    user = UserService.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system."
        )
    user = UserService.create(db, user_in=user_in)
    return user


@router.get("/me", response_model=User)
async def read_current_user(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current user data
    """
    logger.info(f"Retrieved information for user: {current_user.email}")
    return current_user
