#!/usr/bin/env python3
"""
Authentication core module.

This module provides core authentication functionality including
token validation, user authentication, and superuser access control.
"""

from typing import Optional
from datetime import datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer as OAuth2Bearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.core.database import get_db
from app.services.user import UserService
from app.models.user import User

logger = get_logger("auth")

oauth2_scheme = OAuth2Bearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Validate access token and return current user.

    Args:
        db: Database session
        token: JWT access token

    Returns:
        User: Current authenticated user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )
        email: str = payload.get("sub")
        if email is None:
            logger.error("Token payload missing subject")
            raise credentials_exception

        # Check token expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcfromtimestamp(exp) < datetime.utcnow():
            logger.error("Token has expired")
            raise credentials_exception

    except JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise credentials_exception

    user = UserService.get_by_email(db, email)
    if user is None:
        logger.error(f"User not found: {email}")
        raise credentials_exception

    if not user.is_active:
        logger.error(f"Inactive user attempted access: {email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    logger.info(f"Authenticated user: {email}")
    return user


def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Validate that the current user is an active superuser.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current superuser

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        logger.error(f"Non-superuser attempted privileged access: {current_user.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user
