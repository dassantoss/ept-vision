#!/usr/bin/env python3
"""Authentication testing script.

This script performs integration tests for user authentication:
- User creation
- Password hashing
- Authentication validation
- Error handling

Tests include:
- Successful user creation
- Successful authentication
- Failed authentication with wrong password
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.schemas.base import UserCreate
from app.services.user import UserService
from app.core.logging import get_logger

logger = get_logger("test_auth")


def create_test_user(db: Session) -> bool:
    """Create a test user in the database.

    Args:
        db: Database session

    Returns:
        True if user creation successful

    Raises:
        Exception: If user creation fails
    """
    try:
        test_user = UserCreate(
            email="test@example.com",
            password="testpassword123",
            full_name="Test User"
        )

        user = UserService.create(db=db, user_in=test_user)
        logger.info(f"✅ User created: {user.email}")
        return True

    except Exception as e:
        logger.error(f"❌ Error creating test user: {str(e)}")
        raise


def test_valid_authentication(db: Session) -> bool:
    """Test authentication with valid credentials.

    Args:
        db: Database session

    Returns:
        True if authentication successful
    """
    authenticated_user = UserService.authenticate(
        db=db,
        email="test@example.com",
        password="testpassword123"
    )

    if authenticated_user:
        logger.info("✅ Authentication successful")
        return True
    else:
        logger.error("❌ Authentication failed")
        return False


def test_invalid_authentication(db: Session) -> bool:
    """Test authentication with invalid credentials.

    Args:
        db: Database session

    Returns:
        True if authentication correctly rejected
    """
    authenticated_user = UserService.authenticate(
        db=db,
        email="test@example.com",
        password="wrongpassword"
    )

    if not authenticated_user:
        logger.info("✅ Authentication correctly rejected")
        return True
    else:
        logger.error("❌ Error: authentication should have failed")
        return False


def run_auth_tests() -> None:
    """Run the complete authentication test suite.

    Executes all authentication tests in sequence:
    1. User creation
    2. Valid authentication
    3. Invalid authentication

    Handles database session and cleanup.

    Raises:
        Exception: If any test fails
    """
    db = SessionLocal()
    try:
        logger.info("🧪 Starting authentication tests...")

        # 1. Create test user
        logger.info("\n1️⃣ Creating test user...")
        if not create_test_user(db):
            raise Exception("Failed to create test user")

        # 2. Test valid authentication
        logger.info("\n2️⃣ Testing valid authentication...")
        if not test_valid_authentication(db):
            raise Exception("Valid authentication test failed")

        # 3. Test invalid authentication
        logger.info("\n3️⃣ Testing invalid authentication...")
        if not test_invalid_authentication(db):
            raise Exception("Invalid authentication test failed")

        logger.info("\n✅ All authentication tests passed successfully")

    except Exception as e:
        logger.error(f"❌ Error during tests: {str(e)}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_auth_tests()
