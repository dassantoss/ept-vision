import json
import pytest
from unittest.mock import Mock, patch
from app.core.cache import Cache, ModelCache
from app.core.logging import get_logger

logger = get_logger("ept_vision.test_cache")

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    with patch('redis.Redis') as mock:
        # Configurar el mock para devolver una instancia de Mock
        mock_instance = Mock()
        mock.return_value = mock_instance
        
        # Configurar comportamiento del mock
        mock_instance.get.return_value = None
        mock_instance.setex.return_value = True
        mock_instance.delete.return_value = True
        
        yield mock_instance

@pytest.fixture
def cache(mock_redis):
    """Create a cache instance with mock Redis."""
    return Cache(prefix="test", ttl=60)

@pytest.fixture
def model_cache(mock_redis):
    """Create a model cache instance with mock Redis."""
    return ModelCache()

def test_cache_initialization(mock_redis):
    """Test cache initialization with custom prefix and TTL."""
    cache = Cache(prefix="test", ttl=60)
    assert cache.prefix == "test"
    assert cache.default_ttl == 60
    assert mock_redis.ping.called

def test_cache_set(cache, mock_redis):
    """Test setting values in cache."""
    key = "test_key"
    value = {"data": "test_value"}
    
    cache.set(key, value)
    mock_redis.setex.assert_called_once_with(
        "test:test_key",
        60,
        json.dumps(value).encode()
    )

def test_cache_get(cache, mock_redis):
    """Test getting values from cache."""
    key = "test_key"
    value = {"data": "test_value"}
    
    # Mock cache hit
    mock_redis.get.return_value = json.dumps(value).encode()
    result = cache.get(key)
    assert result == value
    mock_redis.get.assert_called_once_with("test:test_key")
    
    # Mock cache miss
    mock_redis.get.return_value = None
    result = cache.get("nonexistent_key")
    assert result is None

def test_cache_delete(cache, mock_redis):
    """Test deleting values from cache."""
    key = "test_key"
    
    cache.delete(key)
    mock_redis.delete.assert_called_once_with("test:test_key")

def test_model_cache_initialization(mock_redis):
    """Test model cache initialization."""
    cache = ModelCache()
    assert cache.prefix == "model"
    assert cache.default_ttl == 3600
    assert mock_redis.ping.called

def test_model_cache_prediction(model_cache, mock_redis):
    """Test caching model predictions."""
    key = "test_image.jpg"
    prediction = {"class": "dog", "confidence": 0.95}
    
    # Test cache miss with new prediction
    mock_redis.get.return_value = None
    model_cache.set(key, prediction)
    mock_redis.setex.assert_called_once_with(
        "model:test_image.jpg",
        3600,
        json.dumps(prediction).encode()
    )
    
    # Test cache hit
    mock_redis.get.return_value = json.dumps(prediction).encode()
    result = model_cache.get(key)
    assert result == prediction
    mock_redis.get.assert_called_with("model:test_image.jpg") 