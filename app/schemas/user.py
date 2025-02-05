#!/usr/bin/env python3
"""User Pydantic models for request/response handling."""

from typing import Optional
from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Base user schema with required fields.

    Attributes:
        email: User's email address
        full_name: User's full name
        is_superuser: Admin privileges flag
        is_active: Account status flag
    """
    email: EmailStr
    full_name: str
    is_superuser: bool = False
    is_active: bool = True


class UserCreate(UserBase):
    """Schema for user creation requests.

    Attributes:
        password: User's password
    """
    password: str


class UserUpdate(UserBase):
    """Schema for user update requests.

    Attributes:
        password: Optional new password
    """
    password: Optional[str] = None


class User(UserBase):
    """Schema for user responses.

    Attributes:
        id: User's unique identifier
    """
    id: int

    class Config:
        from_attributes = True
