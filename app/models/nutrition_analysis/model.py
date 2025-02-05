#!/usr/bin/env python3
"""
Nutrition analysis model implementation.

This module implements a Vision Transformer (ViT) based model for analyzing
nutritional states in animals. It includes functionality for BMI estimation,
nutritional state classification, and providing dietary recommendations.
"""

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    ViTForImageClassification,
    ViTImageProcessor
)
from app.models.base_model import BaseMLModel
from app.core.model_config import NUTRITION_ANALYSIS_CONFIG
from app.utils.model_utils import (
    postprocess_predictions,
    load_model_weights,
    get_device
)
from app.core.logging import get_logger


logger = get_logger("ept_vision.nutrition_analysis_model")


class NutritionAnalysisConfig:
    """
    Configuration for nutrition analysis model.

    This class defines all configuration parameters for the nutrition
    analysis model, including model paths, thresholds, and nutritional
    state definitions.

    Attributes:
        weights_path: Path to model weights file
        image_size: Input image dimensions (height, width)
        num_classes: Number of nutritional states to classify
        confidence_thresholds: Thresholds for confidence levels
        nutrition_states: Mapping of class indices to nutritional states
        bmi_ranges: BMI range definitions for each nutritional state
    """

    weights_path = "app/models/nutrition_analysis/weights/model.pt"
    image_size = (224, 224)
    num_classes = 4  # underweight, normal, overweight, obese
    confidence_thresholds = {
        "high": 0.85,
        "medium": 0.65,
        "low": 0.5
    }
    nutrition_states = {
        0: "underweight",
        1: "normal",
        2: "overweight",
        3: "obese"
    }
    bmi_ranges = {
        "underweight": (0, 18.5),
        "normal": (18.5, 24.9),
        "overweight": (25, 29.9),
        "obese": (30, float('inf'))
    }


class NutritionAnalysisModel(BaseMLModel):
    """
    Nutrition analysis model implementation.

    This class implements a Vision Transformer based model for analyzing
    nutritional states in animals. It handles the complete pipeline from
    image preprocessing to prediction and nutritional recommendations.

    Attributes:
        config: Model configuration instance
        device: Computation device (CPU/GPU)
        processor: ViT image processor for preprocessing
        model: Loaded ViT model instance
    """

    def __init__(self):
        """Initialize the nutrition analysis model."""
        self.config = NutritionAnalysisConfig()
        self.device = get_device()
        logger.info("Initializing nutrition analysis model...")

        try:
            self.processor = ViTImageProcessor.from_pretrained(
                'google/vit-base-patch16-224'
            )
            logger.info("ViT processor initialized successfully")
            super().__init__(model_path=self.config.weights_path)
            logger.info("Nutrition analysis model initialized successfully")
        except Exception as e:
            logger.error(
                "Failed to initialize nutrition analysis model: %s",
                str(e),
                exc_info=True
            )
            raise

    def load_model(self) -> nn.Module:
        """
        Load the ViT model.

        Returns:
            nn.Module: Loaded and configured ViT model

        Raises:
            Exception: If model loading fails
        """
        try:
            model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=self.config.num_classes,
                ignore_mismatched_sizes=True
            )

            try:
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device
                )
                model_dict = model.state_dict()
                state_dict = {
                    k: v for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded custom weights from %s", self.model_path)
            except FileNotFoundError:
                logger.warning(
                    "Custom weights not found at %s, using pretrained weights",
                    self.model_path
                )
            except Exception as e:
                logger.warning(
                    "Error loading custom weights: %s. Using pretrained weights",
                    str(e)
                )

            model = model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error(
                "Failed to load nutrition analysis model: %s",
                str(e),
                exc_info=True
            )
            raise

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image.

        Args:
            image: PIL Image to process

        Returns:
            torch.Tensor: Preprocessed image tensor

        Raises:
            ValueError: If input is not a PIL Image
            Exception: If preprocessing fails
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")

        try:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs.pixel_values.to(self.device)
        except Exception as e:
            logger.error("Error preprocessing image: %s", str(e))
            raise

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Make a prediction on the preprocessed input.

        Args:
            input_tensor: Preprocessed image tensor

        Returns:
            Dict[str, float]: Raw prediction results with confidence scores

        Raises:
            Exception: If prediction fails
        """
        try:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
        except Exception as e:
            logger.error("Error making prediction: %s", str(e))
            raise

    def get_confidence_level(self, confidence: float) -> str:
        """
        Get the confidence level based on the prediction confidence.

        Args:
            confidence: Prediction confidence score

        Returns:
            str: Confidence level category
        """
        if confidence >= self.config.confidence_thresholds["high"]:
            return "high"
        elif confidence >= self.config.confidence_thresholds["medium"]:
            return "medium"
        elif confidence >= self.config.confidence_thresholds["low"]:
            return "low"
        else:
            return "very low"

    def estimate_bmi(self, nutrition_state: str) -> Tuple[float, float]:
        """
        Estimate BMI range based on nutrition state.

        Args:
            nutrition_state: Predicted nutritional state

        Returns:
            Tuple[float, float]: Estimated BMI range (min, max)
        """
        return self.config.bmi_ranges[nutrition_state]

    def get_nutrition_recommendations(self, nutrition_state: str) -> str:
        """
        Get nutrition recommendations based on the predicted state.

        Args:
            nutrition_state: Predicted nutritional state

        Returns:
            str: Dietary and health recommendations
        """
        recommendations = {
            "underweight": (
                "Increase caloric intake with nutrient-rich foods. "
                "Focus on protein and healthy fats."
            ),
            "normal": (
                "Maintain current diet with balanced nutrition. "
                "Regular exercise recommended."
            ),
            "overweight": (
                "Reduce caloric intake moderately. "
                "Increase physical activity."
            ),
            "obese": (
                "Consult healthcare provider. "
                "Focus on balanced diet and regular exercise."
            )
        }
        return recommendations.get(
            nutrition_state,
            "Consult a healthcare provider for personalized advice."
        )

    def postprocess(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Postprocess the model predictions.

        Args:
            predictions: Raw model predictions

        Returns:
            Dict[str, Any]: Processed results with interpretations

        Raises:
            Exception: If postprocessing fails
        """
        try:
            predicted_class = predictions["predicted_class"]
            confidence = predictions["confidence"]

            nutrition_state = self.config.nutrition_states[predicted_class]
            confidence_level = self.get_confidence_level(confidence)
            bmi_range = self.estimate_bmi(nutrition_state)
            recommendations = self.get_nutrition_recommendations(
                nutrition_state
            )

            return {
                "nutrition_state": nutrition_state,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "estimated_bmi_range": bmi_range,
                "recommendations": recommendations
            }
        except Exception as e:
            logger.error("Error postprocessing predictions: %s", str(e))
            raise

    def __call__(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image through the full pipeline.

        Args:
            image: Input PIL Image

        Returns:
            Dict[str, Any]: Complete analysis results

        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            preprocessed = self.preprocess(image)
            predictions = self.predict(preprocessed)
            return self.postprocess(predictions)
        except Exception as e:
            logger.error("Error processing image through pipeline: %s", str(e))
            raise
