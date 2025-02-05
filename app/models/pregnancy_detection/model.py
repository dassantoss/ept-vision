#!/usr/bin/env python3
"""
Pregnancy detection model implementation.

This module implements a combined EfficientNet and SegFormer based model for
detecting pregnancy indicators in animals. It includes functionality for body
segmentation, morphological analysis, and pregnancy state classification.
"""

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_b0
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor
)
from dataclasses import dataclass
from enum import Enum

from app.models.base_model import BaseMLModel
from app.core.model_config import PREGNANCY_DETECTION_CONFIG
from app.utils.model_utils import get_device
from app.core.logging import get_logger

logger = get_logger("ept_vision.pregnancy_detection_model")


class PregnancyIndicator(str, Enum):
    """
    Enumeration of pregnancy indicator states.

    Attributes:
        NONE: No pregnancy indicators detected
        POSSIBLE: Possible pregnancy indicators present
        VISIBLE: Clear pregnancy indicators visible
    """
    NONE = "none"
    POSSIBLE = "possible"
    VISIBLE = "visible"


@dataclass
class AbdominalMetrics:
    """
    Metrics from abdominal analysis.

    Attributes:
        width_ratio: Ratio of abdominal width to total width
        height_ratio: Ratio of abdominal height to total height
        area_ratio: Ratio of abdominal area to total area
        symmetry_score: Score indicating abdominal symmetry
        contour_smoothness: Score indicating contour smoothness
    """
    width_ratio: float
    height_ratio: float
    area_ratio: float
    symmetry_score: float
    contour_smoothness: float


class PregnancyDetectionConfig:
    """
    Configuration for pregnancy detection model.

    This class defines all configuration parameters for the pregnancy detection
    model, including model paths, thresholds, and morphological metrics.

    Attributes:
        weights_path: Path to model weights file
        segformer_model: Name of pretrained SegFormer model
        image_size: Input image dimensions (height, width)
        num_classes: Number of classification states
        confidence_thresholds: Thresholds for confidence levels
        pregnancy_thresholds: Thresholds for pregnancy indicators
    """

    def __init__(self):
        """Initialize pregnancy detection configuration."""
        self.weights_path = (
            "app/models/pregnancy_detection/weights/model.pt"
        )
        self.segformer_model = (
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        self.image_size = (512, 512)  # Increased for better segmentation
        self.num_classes = 2

        # Adjusted thresholds for higher sensitivity
        self.confidence_thresholds = {
            "high": 0.45,    # Lowered from 0.65
            "medium": 0.25,  # Lowered from 0.45
            "low": 0.15      # Lowered from 0.25
        }

        # Pregnancy classification metrics
        self.pregnancy_thresholds = {
            "width_ratio": {
                "possible": 0.20,  # Lowered from 0.30
                "visible": 0.30    # Lowered from 0.40
            },
            "area_ratio": {
                "possible": 0.15,  # Lowered from 0.20
                "visible": 0.25    # Lowered from 0.30
            },
            "combined_confidence": {
                "possible": 0.25,  # Lowered from 0.45
                "visible": 0.40    # Lowered from 0.60
            }
        }


class PregnancyDetectionModel(BaseMLModel):
    """
    Pregnancy detection model implementation.

    This class implements a combined EfficientNet and SegFormer based model
    for detecting pregnancy indicators. It handles the complete pipeline from
    image preprocessing to prediction and morphological analysis.

    Attributes:
        config: Model configuration instance
        device: Computation device (CPU/GPU)
        seg_processor: SegFormer image processor
        seg_model: SegFormer model for body segmentation
        transform: Image transformation pipeline
    """

    def __init__(self):
        """Initialize the pregnancy detection model."""
        self.config = PregnancyDetectionConfig()
        self.device = get_device()
        super().__init__(model_path=self.config.weights_path)

        # Initialize segmentation processor and model
        self.seg_processor = SegformerImageProcessor.from_pretrained(
            self.config.segformer_model
        )
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
            self.config.segformer_model
        )
        self.seg_model.to(self.device)
        self.seg_model.eval()

        # Transform pipeline for main model
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("Enhanced pregnancy detection model initialized successfully")

    def load_model(self) -> nn.Module:
        """
        Load the EfficientNet model.

        Returns:
            nn.Module: Loaded and configured EfficientNet model

        Raises:
            Exception: If model loading fails
        """
        try:
            model = efficientnet_b0(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1280, self.config.num_classes)
            )

            try:
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device
                )
                model.load_state_dict(state_dict)
                logger.info("Loaded custom weights from %s", self.model_path)
            except FileNotFoundError:
                logger.warning(
                    "Custom weights not found at %s, using pretrained weights",
                    self.model_path
                )

            model = model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error(
                "Failed to load pregnancy detection model: %s",
                str(e)
            )
            raise

    def segment_body(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """
        Segment the animal body using SegFormer.

        Args:
            image: Input PIL Image

        Returns:
            Tuple containing:
                - Binary segmentation mask as numpy array
                - Segmentation confidence score

        Raises:
            Exception: If segmentation fails
        """
        try:
            # Prepare image for segmentation
            inputs = self.seg_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.seg_model(**inputs)
                logits = outputs.logits
                seg_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

                # Convert to binary mask (animal vs background)
                binary_mask = (seg_mask > 0).astype(np.uint8)

                # Calculate segmentation confidence
                seg_confidence = torch.softmax(logits, dim=1).max().item()

                return binary_mask, seg_confidence

        except Exception as e:
            logger.error(f"Error in body segmentation: {str(e)}")
            raise

    def analyze_abdominal_region(self, mask: np.ndarray) -> AbdominalMetrics:
        """
        Analyze abdominal region using segmentation mask.

        Args:
            mask: Binary segmentation mask

        Returns:
            AbdominalMetrics: Calculated morphological metrics

        Raises:
            ValueError: If no contours found in mask
            Exception: If analysis fails
        """
        try:
            # Find body contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                raise ValueError("No contours found in segmentation mask")

            # Get main contour
            main_contour = max(contours, key=cv2.contourArea)

            # Get bounding box and measurements
            x, y, w, h = cv2.boundingRect(main_contour)
            total_area = mask.shape[0] * mask.shape[1]
            body_area = cv2.contourArea(main_contour)

            # Divide body into regions
            body_height = h
            abdomen_region = mask[
                y + body_height//3:y + 2*body_height//3,
                x:x+w
            ]

            # Calculate metrics
            width_ratio = w / mask.shape[1]
            height_ratio = h / mask.shape[0]
            area_ratio = body_area / total_area

            # Calculate symmetry
            left_half = abdomen_region[:, :abdomen_region.shape[1]//2]
            right_half = cv2.flip(
                abdomen_region[:, abdomen_region.shape[1]//2:],
                1
            )
            symmetry_score = np.mean(left_half == right_half)

            # Calculate contour smoothness
            perimeter = cv2.arcLength(main_contour, True)
            smoothness = 4 * np.pi * body_area / (perimeter * perimeter)

            return AbdominalMetrics(
                width_ratio=width_ratio,
                height_ratio=height_ratio,
                area_ratio=area_ratio,
                symmetry_score=symmetry_score,
                contour_smoothness=smoothness
            )

        except Exception as e:
            logger.error(f"Error analyzing abdominal region: {str(e)}")
            raise

    def evaluate_pregnancy_indicators(
        self,
        metrics: AbdominalMetrics,
        base_confidence: float
    ) -> Tuple[PregnancyIndicator, float]:
        """
        Evaluate pregnancy indicators based on morphological metrics.

        Args:
            metrics: Calculated abdominal metrics
            base_confidence: Base model confidence score

        Returns:
            Tuple containing:
                - PregnancyIndicator enum value
                - Combined confidence score

        Raises:
            Exception: If evaluation fails
        """
        try:
            # Calculate combined score
            morphology_score = (
                metrics.width_ratio * 0.3 +
                metrics.area_ratio * 0.3 +
                metrics.symmetry_score * 0.2 +
                metrics.contour_smoothness * 0.2
            )

            # Combine with base model confidence
            combined_confidence = (base_confidence + morphology_score) / 2

            # Evaluate indicators
            width_visible = (
                metrics.width_ratio >=
                self.config.pregnancy_thresholds["width_ratio"]["visible"]
            )
            area_visible = (
                metrics.area_ratio >=
                self.config.pregnancy_thresholds["area_ratio"]["visible"]
            )
            conf_visible = (
                combined_confidence >=
                self.config.pregnancy_thresholds["combined_confidence"]["visible"]
            )

            if width_visible and area_visible and conf_visible:
                return PregnancyIndicator.VISIBLE, combined_confidence

            width_possible = (
                metrics.width_ratio >=
                self.config.pregnancy_thresholds["width_ratio"]["possible"]
            )
            area_possible = (
                metrics.area_ratio >=
                self.config.pregnancy_thresholds["area_ratio"]["possible"]
            )
            conf_possible = (
                combined_confidence >=
                self.config.pregnancy_thresholds["combined_confidence"]["possible"]
            )

            if width_possible and area_possible and conf_possible:
                return PregnancyIndicator.POSSIBLE, combined_confidence

            return PregnancyIndicator.NONE, combined_confidence

        except Exception as e:
            logger.error(f"Error evaluating pregnancy indicators: {str(e)}")
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
            tensor = self.transform(image).unsqueeze(0)
            return tensor.to(self.device)
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
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

                return {
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
        except Exception as e:
            logger.error("Error making prediction: %s", str(e))
            raise

    def postprocess(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the model predictions.

        Args:
            predictions: Raw model predictions

        Returns:
            Dict[str, Any]: Processed results with pregnancy indicators
                           and confidence analysis

        Raises:
            Exception: If postprocessing fails
        """
        try:
            predicted_class = predictions["predicted_class"]
            confidence = predictions["confidence"]

            # Determine confidence level
            if confidence >= self.config.confidence_thresholds["high"]:
                confidence_level = "high"
            elif confidence >= self.config.confidence_thresholds["medium"]:
                confidence_level = "medium"
            elif confidence >= self.config.confidence_thresholds["low"]:
                confidence_level = "low"
            else:
                confidence_level = "very low"

            # Convert predicted class to pregnancy indicator
            pregnancy_status = (
                PregnancyIndicator.VISIBLE
                if predicted_class == 1
                else PregnancyIndicator.NONE
            )

            metrics = {
                "width_ratio": getattr(self, '_last_metrics', None),
                "seg_confidence": getattr(self, '_last_seg_confidence', None)
            }

            return {
                "predicted_class": pregnancy_status,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Error postprocessing predictions: {str(e)}")
            raise

    def __call__(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image through the enhanced pipeline.

        Args:
            image: Input PIL Image

        Returns:
            Dict[str, Any]: Complete analysis results including pregnancy
                           indicators and morphological metrics

        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            # 1. Body segmentation
            body_mask, seg_confidence = self.segment_body(image)
            self._last_seg_confidence = seg_confidence

            # Return early if segmentation is not confident
            if seg_confidence < 0.5:
                logger.warning(f"Low segmentation confidence: {seg_confidence:.3f}")
                return {
                    "predicted_class": PregnancyIndicator.NONE,
                    "confidence": 0.0,
                    "confidence_level": "very low",
                    "metrics": {
                        "width_ratio": None,
                        "seg_confidence": seg_confidence
                    }
                }

            # 2. Morphological analysis
            metrics = self.analyze_abdominal_region(body_mask)
            self._last_metrics = metrics

            # 3. Base model prediction
            preprocessed = self.preprocess(image)
            predictions = self.predict(preprocessed)

            # 4. Combined evaluation
            pregnancy_status, combined_confidence = (
                self.evaluate_pregnancy_indicators(
                    metrics,
                    predictions["confidence"]
                )
            )

            predictions["confidence"] = combined_confidence
            return self.postprocess(predictions)

        except Exception as e:
            logger.error(f"Error processing image through pipeline: {str(e)}")
            raise
