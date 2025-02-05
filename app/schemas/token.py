#!/usr/bin/env python3
"""Authentication token schemas."""

from typing import Optional
from pydantic import BaseModel


class Token(BaseModel):
    """Schema for authentication token response.

    Attributes:
        access_token: JWT access token
        token_type: Token type (e.g., 'bearer')
    """
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    """Schema for token payload data.

    Attributes:
        sub: Subject identifier (user ID)
    """
    sub: Optional[int] = None
