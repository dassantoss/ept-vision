#!/usr/bin/env python3
"""
Authentication endpoints for the API.

This module handles user authentication operations including login
and current user information retrieval.
"""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_access_token
from app.core.auth import get_current_user
from app.schemas.token import Token
from app.schemas.user import User
from app.services.user import UserService
from app.core.logging import get_logger


logger = get_logger("ept_vision.auth")
router = APIRouter()


@router.post("/login")
async def login(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> dict[str, Any]:
    """
    OAuth2 compatible token login endpoint.

    Args:
        db: Database session
        form_data: OAuth2 password request form data

    Returns:
        dict: Access token and token type

    Raises:
        HTTPException: If credentials are invalid or user is inactive
    """
    user = UserService.authenticate(
        db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    elif not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
        )

    access_token_expires = timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    access_token = create_access_token(
        user.id, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/me", response_model=User)
async def read_current_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current user information
    """
    logger.info(f"Retrieved user information for user: {current_user.email}")
    return current_user
