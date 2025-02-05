#!/usr/bin/env python3
"""Base class for machine learning models implementation.

This module provides a base abstract class that defines the common interface
and functionality for all ML models in the application.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from app.core.cache import model_cache


class BaseMLModel(ABC):
    """Abstract base class for ML models.

    Provides common functionality for model initialization, caching,
    and prediction pipeline. All ML models should inherit from this class.

    Attributes:
        model_path: Path to model weights file
        device: Device to run model on (CPU/GPU)
        model: The actual ML model instance
    """

    def __init__(self, model_path: str):
        """Initialize base ML model.

        Args:
            model_path: Path to the model weights file
        """
        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize model after attributes are set."""
        self.model = self.load_model()

    @property
    def model_name(self) -> str:
        """Get model name for caching.

        Returns:
            String containing model class name
        """
        return self.__class__.__name__

    def get_cached_prediction(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction for an image.

        Args:
            image_id: Unique identifier for the image

        Returns:
            Cached prediction if exists, None otherwise
        """
        return model_cache.get_prediction(self.model_name, image_id)

    def cache_prediction(
        self,
        image_id: str,
        prediction: Dict[str, Any],
        ttl: int = None
    ) -> None:
        """Cache a prediction result.

        Args:
            image_id: Unique identifier for the image
            prediction: Prediction result to cache
            ttl: Time to live for cache entry in seconds
        """
        model_cache.set_prediction(self.model_name, image_id, prediction, ttl)

    def predict_with_cache(
        self,
        image: Image.Image,
        image_id: str
    ) -> Dict[str, Any]:
        """Get prediction with caching support.

        Args:
            image: Input PIL image to process
            image_id: Unique identifier for the image

        Returns:
            Dictionary containing prediction results
        """
        cached_prediction = self.get_cached_prediction(image_id)
        if cached_prediction:
            return cached_prediction

        processed_image = self.preprocess(image)
        predictions = self.predict(processed_image)
        result = self.postprocess(predictions)

        self.cache_prediction(image_id, result)

        return result

    @abstractmethod
    def load_model(self) -> Any:
        """Load model weights and initialize model.

        Returns:
            Initialized model instance
        """
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess input image for model.

        Args:
            image: Input PIL image

        Returns:
            Preprocessed image tensor
        """
        pass

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run prediction on preprocessed input.

        Args:
            input_tensor: Preprocessed input tensor

        Returns:
            Dictionary containing raw prediction results
        """
        pass

    @abstractmethod
    def postprocess(self, predictions: Any) -> Dict[str, Any]:
        """Process raw predictions into final format.

        Args:
            predictions: Raw prediction outputs

        Returns:
            Dictionary containing processed results
        """
        pass
