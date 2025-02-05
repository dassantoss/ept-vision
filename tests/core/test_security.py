import pytest
from datetime import timedelta
from jose import jwt

from app.core.security import create_access_token, verify_password, get_password_hash
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("test_security")

def test_password_hashing():
    """Test password hashing and verification."""
    password = "testpassword123"
    
    # Test hashing
    hashed = get_password_hash(password)
    assert hashed != password
    assert len(hashed) > 0
    
    # Test verification
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False

def test_access_token_creation():
    """Test JWT token creation and validation."""
    # Test with default expiration
    subject = "test@example.com"
    token = create_access_token(subject)
    
    # Decode and verify token
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    assert payload["sub"] == subject
    assert "exp" in payload
    
    # Test with custom expiration
    expires = timedelta(minutes=5)
    token = create_access_token(subject, expires)
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    assert payload["sub"] == subject

def test_token_expiration():
    """Test token expiration."""
    subject = "test@example.com"
    expires = timedelta(seconds=1)
    token = create_access_token(subject, expires)
    
    # Token should be valid initially
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    assert payload["sub"] == subject
    
    # Wait for token to expire
    import time
    time.sleep(2)
    
    # Token should be expired
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])

def test_invalid_token():
    """Test invalid token handling."""
    # Test with wrong secret key
    subject = "test@example.com"
    token = create_access_token(subject)
    
    with pytest.raises(jwt.JWTError):
        jwt.decode(token, "wrong_secret", algorithms=["HS256"])
    
    # Test with tampered token
    with pytest.raises(jwt.JWTError):
        jwt.decode(token + "tampered", settings.SECRET_KEY, algorithms=["HS256"]) 