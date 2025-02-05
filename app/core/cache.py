#!/usr/bin/env python3
"""Cache module for managing in-memory caching functionality.

This module provides caching capabilities through the Cache and
ModelCache classes, allowing for efficient storage and retrieval
of data in memory.
"""

import json
from typing import Any, Dict, Optional, Union
from redis import Redis
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("cache")


class Cache:
    """
    Base cache implementation for general purpose caching.

    Attributes:
        redis_client: Redis connection instance
        prefix: Key prefix for namespacing
        default_ttl: Default time-to-live for cache entries
    """
    def __init__(self, prefix: str = "", ttl: int = 3600):
        """
        Initialize cache with optional prefix and TTL.

        Args:
            prefix: Key prefix for namespacing
            ttl: Default time-to-live in seconds
        """
        self.redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True
        )
        self.prefix = prefix
        self.default_ttl = ttl
        logger.info(f"Cache initialized with prefix '{prefix}' and TTL {ttl}s")

    def _get_key(self, key: str) -> str:
        """
        Generate a prefixed cache key.

        Args:
            key: Original cache key

        Returns:
            str: Prefixed cache key
        """
        return f"{self.prefix}:{key}" if self.prefix else key

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store a value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to store
            ttl: Optional time-to-live in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            key = self._get_key(key)
            result = self.redis_client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(value)
            )
            logger.debug(f"Cache SET: {key} (TTL: {ttl or self.default_ttl}s)")
            return result
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.

        Args:
            key: Cache key

        Returns:
            Optional[Any]: Cached value if found, None otherwise
        """
        try:
            key = self._get_key(key)
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return None

    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            key = self._get_key(key)
            result = bool(self.redis_client.delete(key))
            if result:
                logger.debug(f"Cache DELETE: {key}")
            return result
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False


class ModelCache(Cache):
    """
    Specialized cache implementation for ML model predictions.

    Extends the base Cache class with methods specific to handling
    model predictions.
    """
    def __init__(self):
        """Initialize model cache with predefined prefix and TTL."""
        super().__init__(prefix="model", ttl=3600)
        logger.info("ModelCache initialized")

    def get_prediction(self, model_name: str, image_id: str) -> Optional[dict]:
        """
        Get cached prediction for a specific model and image.

        Args:
            model_name: Name of the model
            image_id: ID of the image

        Returns:
            Optional[dict]: Cached prediction if found, None otherwise
        """
        key = f"{model_name}:{image_id}"
        return self.get(key)

    def set_prediction(
        self,
        model_name: str,
        image_id: str,
        prediction: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a prediction result for a specific model and image.

        Args:
            model_name: Name of the model
            image_id: ID of the image
            prediction: Prediction results to cache
            ttl: Optional time-to-live in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        key = f"{model_name}:{image_id}"
        return self.set(key, prediction, ttl)

    def delete_prediction(self, model_name: str, image_id: str) -> bool:
        """
        Delete a cached prediction.

        Args:
            model_name: Name of the model
            image_id: ID of the image

        Returns:
            bool: True if successful, False otherwise
        """
        key = f"{model_name}:{image_id}"
        return self.delete(key)


# Singleton instances
cache = Cache()
model_cache = ModelCache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return Cache()
