#!/usr/bin/env python3
"""Base Pydantic models for data validation.

Provides base schemas for user data validation and serialization.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Base schema for user data.

    Attributes:
        email: User's email address
        is_active: Account status flag
        is_superuser: Admin privileges flag
        full_name: User's full name
    """
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """Schema for user creation.

    Attributes:
        email: Required email address
        password: User's password
    """
    email: EmailStr
    password: str


class UserInDBBase(UserBase):
    """Base schema for user database representation.

    Attributes:
        id: User's unique identifier
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
