import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app
from app.core.logging import get_logger
from app.core.security import create_access_token, get_password_hash
from app.schemas.base import UserCreate

logger = get_logger("test_auth")

client = TestClient(app)

@pytest.fixture
def test_user():
    return {
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpassword123"
    }

@pytest.fixture
def mock_user_service():
    with patch('app.api.v1.endpoints.auth.UserService') as mock:
        mock_user = Mock()
        mock_user.email = "test@example.com"
        mock_user.is_active = True
        mock_user.hashed_password = get_password_hash("testpassword123")
        mock.get_by_email.return_value = mock_user
        mock.authenticate.return_value = mock_user
        yield mock

def test_login_success(mock_user_service, test_user):
    """Test successful login."""
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user["email"],
            "password": test_user["password"]
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_invalid_credentials(mock_user_service, test_user):
    """Test login with invalid credentials."""
    mock_user_service.authenticate.return_value = None
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user["email"],
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"

def test_login_inactive_user(mock_user_service, test_user):
    """Test login with inactive user."""
    mock_user = mock_user_service.get_by_email.return_value
    mock_user.is_active = False
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user["email"],
            "password": test_user["password"]
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Inactive user"

@pytest.mark.integration
def test_full_auth_flow(mock_user_service, test_user):
    """Test the complete authentication flow."""
    # 1. Login
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user["email"],
            "password": test_user["password"]
        }
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    # 2. Use token to access protected endpoint
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/v1/users/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["email"] == test_user["email"] 