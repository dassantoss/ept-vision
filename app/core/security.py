#!/usr/bin/env python3
"""
Security utilities module.

This module provides core security functionality including password hashing,
verification, and JWT token generation for user authentication and
authorization.
"""

from datetime import datetime, timedelta
from typing import Any, Union

from jose import jwt
from passlib.context import CryptContext
from app.core.logging import get_logger

from app.core.config import settings


logger = get_logger("security")

# Configure the password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any],
    expires_delta: timedelta = None
) -> str:
    """
    Create a JWT access token.

    This function generates a new JWT token for a given subject (typically
    a user ID) with an optional expiration time.

    Args:
        subject: The subject to encode in the token (usually user ID)
        expires_delta: Optional custom expiration time, defaults to 30 minutes

    Returns:
        str: Encoded JWT token

    Example:
        >>> token = create_access_token("user123", timedelta(hours=1))
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)

    to_encode = {"exp": expire, "sub": str(subject)}
    try:
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm="HS256"
        )
        logger.debug(f"Created access token for subject: {subject}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to compare against

    Returns:
        bool: True if password matches, False otherwise

    Example:
        >>> is_valid = verify_password("mypassword", hashed_password)
    """
    try:
        result = pwd_context.verify(plain_password, hashed_password)
        logger.debug(
            f"Password verification {'successful' if result else 'failed'}"
        )
        return result
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False


def get_password_hash(password: str) -> str:
    """
    Generate a hash from a plain text password.

    Args:
        password: The plain text password to hash

    Returns:
        str: The hashed password

    Example:
        >>> hashed = get_password_hash("mypassword")
    """
    try:
        hashed = pwd_context.hash(password)
        logger.debug("Password hash generated successfully")
        return hashed
    except Exception as e:
        logger.error(f"Error generating password hash: {str(e)}")
        raise
