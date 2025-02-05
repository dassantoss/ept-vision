import pytest
from sqlalchemy.orm import Session
from unittest.mock import Mock, patch

from app.services.user import UserService
from app.schemas.user import UserCreate, User
from app.models.user import User
from app.core.logging import get_logger
from app.core.security import get_password_hash, verify_password

logger = get_logger("ept_vision.test_user")

@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return Mock(spec=Session)

@pytest.fixture
def test_user_data():
    """Create test user data."""
    return {
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpassword123",
        "is_superuser": False
    }

@pytest.fixture
def mock_user():
    """Create a mock user with proper hashed password."""
    user = Mock(spec=User)
    user.email = "test@example.com"
    user.full_name = "Test User"
    user.hashed_password = get_password_hash("testpassword123")
    user.is_active = True
    user.is_superuser = False
    return user

def test_create_user(mock_db, test_user_data):
    """Test user creation."""
    user_in = UserCreate(**test_user_data)
    hashed_password = get_password_hash(user_in.password)
    
    # Create a mock user with the expected attributes
    mock_user = Mock()
    mock_user.email = user_in.email
    mock_user.full_name = user_in.full_name
    mock_user.hashed_password = hashed_password
    mock_user.is_superuser = user_in.is_superuser
    mock_user.is_active = True
    
    # Mock the User model
    with patch('app.models.user.User') as mock_user_model:
        mock_user_model.return_value = mock_user
        
        # Create the user
        created_user = UserService.create(mock_db, user_in)
        
        # Verify the user was created correctly
        assert created_user.email == test_user_data["email"]
        assert created_user.full_name == test_user_data["full_name"]
        assert verify_password(test_user_data["password"], created_user.hashed_password)
        assert created_user.is_superuser == test_user_data["is_superuser"]
        assert created_user.is_active
        
        # Verify database operations
        mock_db.add.assert_called_once_with(mock_user)
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(mock_user)

def test_authenticate_user(mock_db, test_user_data):
    """Test user authentication."""
    # Create a mock user
    mock_user = Mock()
    mock_user.email = test_user_data["email"]
    mock_user.hashed_password = get_password_hash(test_user_data["password"])
    mock_user.is_active = True
    
    # Mock the get_by_email method
    with patch('app.services.user.UserService.get_by_email') as mock_get:
        mock_get.return_value = mock_user
        
        # Test successful authentication
        user = UserService.authenticate(
            mock_db,
            email=test_user_data["email"],
            password=test_user_data["password"]
        )
        assert user == mock_user
        
        # Test failed authentication with wrong password
        user = UserService.authenticate(
            mock_db,
            email=test_user_data["email"],
            password="wrongpassword"
        )
        assert user is None
        
        # Test failed authentication with inactive user
        mock_user.is_active = False
        user = UserService.authenticate(
            mock_db,
            email=test_user_data["email"],
            password=test_user_data["password"]
        )
        assert user is None

def test_get_by_email(mock_db, test_user_data):
    """Test getting user by email."""
    # Create a mock user
    mock_user = Mock()
    mock_user.email = test_user_data["email"]
    
    # Mock the database query
    mock_query = Mock()
    mock_query.filter.return_value.first.return_value = mock_user
    mock_db.query.return_value = mock_query
    
    # Test getting existing user
    user = UserService.get_by_email(mock_db, email=test_user_data["email"])
    assert user == mock_user
    
    # Test getting non-existent user
    mock_query.filter.return_value.first.return_value = None
    user = UserService.get_by_email(mock_db, email="nonexistent@example.com")
    assert user is None

@pytest.mark.integration
def test_user_service_integration(mock_db, test_user_data):
    """Test complete user service workflow."""
    # 1. Create user
    user_in = UserCreate(**test_user_data)
    created_user = UserService.create(mock_db, user_in)
    assert created_user.email == test_user_data["email"]
    
    # 2. Get user by email
    retrieved_user = UserService.get_by_email(mock_db, email=test_user_data["email"])
    assert retrieved_user == created_user
    
    # 3. Authenticate user
    authenticated_user = UserService.authenticate(
        mock_db,
        email=test_user_data["email"],
        password=test_user_data["password"]
    )
    assert authenticated_user == created_user 