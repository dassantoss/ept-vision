#!/usr/bin/env python3
"""
Disease detection model implementation.

This module implements a Vision Transformer (ViT) based model for detecting
various diseases and health conditions in animals. It includes configuration,
preprocessing, prediction, and post-processing functionality.
"""

from typing import Any, Dict
import torch
import torch.nn as nn
from PIL import Image
import os

from transformers import ViTForImageClassification, ViTImageProcessor
from app.models.base_model import BaseMLModel
from app.core.logging import get_logger


logger = get_logger("ept_vision.disease_detection_model")


class DiseaseDetectionConfig:
    """
    Configuration for disease detection model.

    This class defines all configuration parameters for the disease detection
    model, including model paths, image specifications, confidence thresholds,
    and disease-specific information.

    Attributes:
        weights_path: Path to model weights file
        image_size: Input image dimensions (height, width)
        num_classes: Number of disease classes to detect
        confidence_thresholds: Thresholds for confidence levels
        class_names: Names of detectable conditions
        severity_levels: Possible severity levels for each condition
        treatment_recommendations: Treatment suggestions by condition/severity
    """

    def __init__(self, weights_path: str = None):
        """
        Initialize disease detection configuration.

        Args:
            weights_path: Optional custom path to model weights
        """
        self.weights_path = (
            weights_path or
            "app/models/disease_detection/weights/model.pt"
        )
        self.image_size = (224, 224)
        self.num_classes = 4  # Adjusted to match specific conditions
        self.confidence_thresholds = {
            "high": 0.45,    # Significantly lowered from 0.65
            "medium": 0.25,  # Significantly lowered from 0.45
            "low": 0.15      # Significantly lowered from 0.25
        }
        # Specific disease states
        self.class_names = [
            "healthy",
            "skin_infection",
            "eye_infection",
            "external_wounds"
        ]
        # Severity levels for each condition
        self.severity_levels = {
            "skin_infection": ["mild", "moderate", "severe"],
            "eye_infection": ["mild", "moderate", "severe"],
            "external_wounds": ["minor", "moderate", "severe"]
        }
        # Treatment recommendations
        self.treatment_recommendations = {
            "skin_infection": {
                "mild": "Clean affected area and apply prescribed topical treatment.",
                "moderate": (
                    "Veterinary consultation required. "
                    "Start antibiotics if prescribed."
                ),
                "severe": (
                    "Immediate veterinary attention needed. "
                    "May require systemic treatment."
                )
            },
            "eye_infection": {
                "mild": "Clean eye area and apply prescribed eye drops.",
                "moderate": "Veterinary examination needed. Follow treatment plan.",
                "severe": (
                    "Emergency veterinary care required. "
                    "Risk of vision complications."
                )
            },
            "external_wounds": {
                "minor": "Clean wound and apply antiseptic. Monitor healing.",
                "moderate": "Veterinary attention needed. May require stitches.",
                "severe": "Emergency care required. Risk of complications."
            }
        }

    def get_class_name(self, class_id: int) -> str:
        """
        Get the class name for a given class ID.

        Args:
            class_id: Integer identifier of the class

        Returns:
            str: Name of the class, defaults to "healthy" if ID is invalid
        """
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return "healthy"  # Default to healthy if invalid class ID


class DiseaseDetectionModel(BaseMLModel):
    """
    Disease detection model implementation.

    This class implements a Vision Transformer based model for detecting
    various diseases and health conditions. It handles the complete
    pipeline from image preprocessing to prediction and result interpretation.

    Attributes:
        config: Model configuration instance
        processor: ViT image processor for preprocessing
        model: Loaded ViT model instance
    """

    def __init__(self, weights_path: str = None):
        """
        Initialize the disease detection model.

        Args:
            weights_path: Optional custom path to model weights
        """
        self.config = DiseaseDetectionConfig(weights_path)
        self.processor = None
        logger.info(
            "Initializing disease detection model with weights path: "
            f"{self.config.weights_path}"
        )

        try:
            # Initialize processor first
            logger.info("Loading ViT processor...")
            self.processor = ViTImageProcessor.from_pretrained(
                'google/vit-base-patch16-224',
                cache_dir='/home/api-user/.cache/huggingface'
            )
            logger.info("ViT processor loaded successfully")

            # Initialize parent class (which loads the model)
            super().__init__(model_path=self.config.weights_path)
            logger.info("Disease detection model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}",
                         exc_info=True)
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
            logger.info("Loading ViT model from pretrained...")
            model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=self.config.num_classes,
                ignore_mismatched_sizes=True,
                cache_dir='/home/api-user/.cache/huggingface'
            )
            logger.info("Base ViT model loaded successfully")

            if os.path.exists(self.model_path):
                try:
                    logger.info(f"Loading custom weights from {self.model_path}")
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
                    logger.info("Custom weights loaded successfully")
                except Exception as e:
                    logger.warning(
                        f"Failed to load custom weights: {str(e)}",
                        exc_info=True
                    )
            else:
                logger.warning(
                    f"Custom weights not found at {self.model_path}, "
                    "using pretrained weights"
                )

            logger.info(f"Moving model to device: {self.device}")
            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}",
                         exc_info=True)
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
            # Convert grayscale to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            inputs = self.processor(images=image, return_tensors="pt")
            return inputs.pixel_values.to(self.device)
        except Exception as e:
            logger.error("Error preprocessing image: %s", str(e))
            raise

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Make a prediction on the preprocessed input.

        Args:
            input_tensor: Preprocessed image tensor

        Returns:
            Dict[str, Any]: Raw prediction results including class
                            probabilities

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
                top_k = torch.topk(probabilities, k=3, dim=1)

                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "top_k_indices": top_k.indices[0].tolist(),
                    "top_k_probabilities": top_k.values[0].tolist(),
                    "probabilities": probabilities[0].tolist()
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

    def estimate_severity(self, condition: str, confidence: float) -> str:
        """
        Estimate severity level based on condition and confidence score.

        Args:
            condition: Detected condition name
            confidence: Prediction confidence score

        Returns:
            str: Estimated severity level
        """
        if condition.lower() == "healthy":
            return "none"

        # Define severity thresholds
        SEVERITY_THRESHOLDS = {
            "infection": {
                (0.8, 1.0): "severe",
                (0.6, 0.8): "moderate",
                (0.0, 0.6): "mild"
            },
            "tumor": {
                (0.8, 1.0): "malignant",
                (0.6, 0.8): "suspicious",
                (0.0, 0.6): "benign"
            }
        }

        condition = condition.lower()
        if condition not in SEVERITY_THRESHOLDS:
            return "unknown"

        thresholds = SEVERITY_THRESHOLDS[condition]
        for (min_conf, max_conf), severity in thresholds.items():
            if min_conf <= confidence <= max_conf:
                return severity

        return "unknown"

    def get_treatment_recommendations(
        self,
        disease_state: str,
        severity: str
    ) -> str:
        """
        Get treatment recommendations based on disease state and severity.

        Args:
            disease_state: Detected disease or condition
            severity: Estimated severity level

        Returns:
            str: Treatment recommendation text
        """
        if disease_state == "healthy":
            return "Continue regular check-ups and preventive care."

        return self.config.treatment_recommendations.get(
            disease_state,
            {}
        ).get(
            severity,
            "Consult with veterinarian for proper diagnosis and treatment."
        )

    def postprocess(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
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
            top_k_indices = predictions["top_k_indices"]
            top_k_probabilities = predictions["top_k_probabilities"]

            disease_state = self.config.get_class_name(predicted_class)
            confidence_level = self.get_confidence_level(confidence)
            severity = self.estimate_severity(disease_state, confidence)
            recommendations = self.get_treatment_recommendations(
                disease_state,
                severity
            )

            # Get top predictions with disease names
            top_predictions = [
                {
                    "condition": self.config.get_class_name(idx),
                    "confidence": prob
                }
                for idx, prob in zip(top_k_indices, top_k_probabilities)
            ]

            return {
                "predicted_class": disease_state,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "severity": severity,
                "recommendations": recommendations,
                "top_predictions": top_predictions
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
