#!/usr/bin/env python3
"""Module for pre-labeling model implementation.

This module provides functionality for automatic pre-labeling of animal images,
including detection of health issues, pregnancy indicators, and body condition.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)
from torchvision import models, transforms
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor
)
from ultralytics import YOLO

from app.core.logging import get_logger
from app.models.disease_detection.model import DiseaseDetectionModel
from app.models.pregnancy_detection.model import PregnancyDetectionModel
from app.models.nutrition_analysis.model import NutritionAnalysisModel

logger = get_logger("ept_vision.prelabeling")

# Definición de mappings al inicio del archivo
LABEL_MAPS = {
    'animal_type': {'dog': 0, 'cat': 1},
    'size': {'small': 0, 'medium': 1, 'large': 2},
    'body_condition': {'underweight': 0, 'normal': 1, 'overweight': 2},
    'visible_health_issues': {'none': 0, 'wounds': 1, 'skin_issues': 2, 'other': 3},
    'pregnancy_indicators': {'none': 0, 'possible': 1, 'visible': 2},
    'image_quality': {'poor': 0, 'medium': 1, 'good': 2},
    'context': {'home': 0, 'street': 1, 'shelter': 2, 'other': 3}
}

@dataclass
class MenuMappings:
    """Menu option mappings for classification results.

    Attributes:
        health_issues: Mapping of health status to menu options
        pregnancy: Mapping of pregnancy status to menu options
        body_condition: Mapping of body condition to menu options
    """
    health_issues: Dict[str, str] = field(default_factory=lambda: {
        'none': 'none',
        'wounds': 'wounds',
        'skin_issues': 'skin_issues',
        'other': 'other'
    })

    pregnancy: Dict[str, str] = field(default_factory=lambda: {
        'none': 'none',
        'possible': 'possible',
        'visible': 'visible'
    })

    body_condition: Dict[str, str] = field(default_factory=lambda: {
        'underweight': 'underweight',
        'normal': 'normal',
        'overweight': 'overweight'
    })


class PreLabelingConfig:
    """Configuration settings for pre-labeling model.

    Handles thresholds, mappings and parameters used in the pre-labeling process.
    Includes settings for animal detection, health analysis, and image quality.
    """
    def __init__(self):
        """Initialize configuration with default values."""
        self.image_size = (224, 224)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = 0.10
        self.discard_threshold = 0.05

        # Significantly lowered thresholds for health and pregnancy detection
        self.health_threshold = 0.05     # Very low threshold for health issues
        self.pregnancy_threshold = 0.05   # Very low threshold for pregnancy detection
        self.body_condition_threshold = 0.08  # Lower threshold for body condition

        # Specific class ranges for ImageNet - Updated with complete cat classes
        self.animal_mapping = {
            'cat': [
                281, 282, 283, 284, 285, 286, 287,  # Cat breeds
                278, 279, 280,                      # Wild cats (tiger cat, Persian cat, etc.)
                268, 269, 270, 271, 272             # Additional cat classes
            ],
            'dog': list(range(151, 269))           # Dog breeds
        }

        # Context mapping based on common ImageNet scenes
        self.context_mapping = {
            'home': ['home', 'indoor', 'room', 'house', 'living', 'sofa', 'couch', 'bed', 'chair'],
            'street': ['street', 'road', 'alley', 'sidewalk', 'outdoor', 'garden', 'yard', 'park'],
            'shelter': ['kennel', 'cage', 'shelter', 'pen', 'fence', 'enclosure'],
            'clinic': ['hospital', 'clinic', 'medical', 'vet', 'laboratory', 'office']
        }

        # Adjusted size thresholds (relative to image area)
        self.size_thresholds = {
            'small': 0.12,    # Less than 12% of image area
            'medium': 0.30,   # Between 12% and 30% of image area
            'large': 1.0      # More than 30% of image area
        }

        # Adjusted quality thresholds
        self.quality_thresholds = {
            'good': 0.60,
            'poor': 0.30
        }

        # Health detection thresholds
        self.health_detection = {
            'edge_density': 0.08,  # Threshold for detecting visible bones/ribs
            'color_variance': 0.15,  # Threshold for skin issues
            'texture_threshold': 0.12  # Threshold for abnormal textures
        }

        # Pregnancy detection thresholds
        self.pregnancy_detection = {
            'width_ratio': 0.35,  # Minimum width ratio for possible pregnancy
            'area_ratio': 0.25,   # Minimum area ratio for possible pregnancy
            'symmetry': 0.70,     # Minimum symmetry score
            'shape_confidence': 0.60  # Minimum shape confidence
        }

        # Mapeo de razas por tamaño
        self.dog_size_mapping = {
            'small': [
                151, 152, 153,  # Chihuahua
                154, 155,       # Japanese Spaniel, Maltese
                156, 157,       # Pekinese, Shih Tzu
                158, 159,       # Toy Terrier, Yorkshire Terrier
                160, 161,       # Miniature Pinscher, Toy Poodle
                162, 163,       # Pug, Brussels Griffon
                164, 165        # Papillon, Miniature Schnauzer
            ],
            'medium': [
                166, 167, 168,  # Border Terrier, Beagle
                169, 170,       # Brittany Spaniel, Cocker Spaniel
                171, 172,       # Welsh Springer Spaniel, English Springer Spaniel
                173, 174,       # Border Collie, Australian Shepherd
                175, 176,       # Shetland Sheepdog, Collie
                177, 178        # Standard Poodle, Australian Terrier
            ],
            'large': [
                179, 180,       # German Shepherd, Doberman
                181, 182,       # Great Dane, Saint Bernard
                183, 184,       # Husky, Alaskan Malamute
                185, 186,       # Bernese Mountain Dog, Rottweiler
                187, 188,       # Golden Retriever, Labrador Retriever
                189, 190        # Newfoundland, Old English Sheepdog
            ]
        }

        # Umbrales de área relativos para validación secundaria
        self.relative_size_thresholds = {
            'small': {
                'min': 0.05,   # Mínimo 5% del área de la imagen
                'max': 0.35    # Máximo 35% del área de la imagen
            },
            'medium': {
                'min': 0.15,   # Mínimo 15% del área de la imagen
                'max': 0.60    # Máximo 60% del área de la imagen
            },
            'large': {
                'min': 0.25,   # Mínimo 25% del área de la imagen
                'max': 0.85    # Máximo 85% del área de la imagen
            }
        }

        # Nuevo: mapeos del menú
        self.menu = MenuMappings()

    def is_animal_class(self, class_idx: int) -> Tuple[bool, str]:
        """Check if class index corresponds to an animal category.

        Args:
            class_idx: ImageNet class index to check

        Returns:
            Tuple containing:
                - Boolean indicating if index is an animal class
                - String indicating animal type ('cat', 'dog', or 'other')
        """
        if class_idx in self.animal_mapping['cat']:
            return True, 'cat'
        elif class_idx in self.animal_mapping['dog']:
            return True, 'dog'
        return False, 'other'


class MultiTaskModel(nn.Module):
    """La misma arquitectura que usamos en el entrenamiento"""
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Task-specific heads
        self.animal_type = nn.Linear(1280, len(LABEL_MAPS['animal_type']))
        self.size = nn.Linear(1280, len(LABEL_MAPS['size']))
        self.body_condition = nn.Linear(1280, len(LABEL_MAPS['body_condition']))
        self.health_issues = nn.Linear(1280, len(LABEL_MAPS['visible_health_issues']))
        self.pregnancy = nn.Linear(1280, len(LABEL_MAPS['pregnancy_indicators']))
        self.image_quality = nn.Linear(1280, len(LABEL_MAPS['image_quality']))
        self.context = nn.Linear(1280, len(LABEL_MAPS['context']))

    def forward(self, x):
        features = self.features(x)
        features = features.mean([2, 3])  # Global average pooling
        
        # Apply softmax to each output
        return {
            'animal_type': F.softmax(self.animal_type(features), dim=1),
            'size': F.softmax(self.size(features), dim=1),
            'body_condition': F.softmax(self.body_condition(features), dim=1),
            'visible_health_issues': F.softmax(self.health_issues(features), dim=1),
            'pregnancy_indicators': F.softmax(self.pregnancy(features), dim=1),
            'image_quality': F.softmax(self.image_quality(features), dim=1),
            'context': F.softmax(self.context(features), dim=1)
        }

class PreLabelingModel:
    """Model for automatic pre-labeling of animal images.

    Performs comprehensive analysis including:
    - Animal type detection
    - Health issue identification
    - Pregnancy indicators
    - Body condition assessment
    - Image quality evaluation
    """

    def __init__(self):
        """Initialize model components and load required weights."""
        self.config = PreLabelingConfig()
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.initialize_model()

        self._disease_model: Optional[DiseaseDetectionModel] = None
        self._pregnancy_model: Optional[PregnancyDetectionModel] = None
        self._nutrition_model: Optional[NutritionAnalysisModel] = None

        self.confidence_threshold = 0.15
        self.size_thresholds = {
            'small': 0.25,
            'medium': 0.50,
            'large': 1.0
        }
        self.default_context = 'home'

        model_path = 'app/models/prelabeling/weights/yolov8x-seg.pt'
        self.yolo_model = YOLO(model_path)

        model_name = "microsoft/resnet-50"
        self.gender_classifier = AutoModelForImageClassification.from_pretrained(
            model_name
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name
        )
        self.vet_model = models.resnet50(pretrained=True)
        self.vet_model.to(self.config.device)
        self.gender_classifier.to(self.config.device)

        logger.info("Models loaded successfully")

    @property
    @lru_cache()
    def disease_model(self) -> DiseaseDetectionModel:
        """Get disease detection model instance."""
        if self._disease_model is None:
            self._disease_model = DiseaseDetectionModel()
        return self._disease_model

    @property
    @lru_cache()
    def pregnancy_model(self) -> PregnancyDetectionModel:
        """Get pregnancy detection model instance."""
        if self._pregnancy_model is None:
            self._pregnancy_model = PregnancyDetectionModel()
        return self._pregnancy_model

    @property
    @lru_cache()
    def nutrition_model(self) -> NutritionAnalysisModel:
        """Get nutrition analysis model instance."""
        if self._nutrition_model is None:
            self._nutrition_model = NutritionAnalysisModel()
        return self._nutrition_model

    def initialize_model(self):
        """Initialize the model and load trained weights."""
        try:
            # Inicializar el modelo
            self.model = MultiTaskModel().to(self.config.device)
            
            # Cargar los weights entrenados
            weights_path = Path(__file__).parent / "weights" / "best_model.pth"
            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=self.config.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                logger.info(f"Loaded custom weights from {weights_path}")
            else:
                raise FileNotFoundError(f"Weights not found at {weights_path}")
                
        except Exception as e:
            logger.error(f"Error initializing pre-labeling model: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            image: Input PIL image

        Returns:
            Preprocessed tensor
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)
            return tensor.to(self.config.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Get model predictions for an image.

        Args:
            image: Input PIL image

        Returns:
            Dictionary of predictions for each task
        """
        try:
            with torch.no_grad():
                tensor = self.preprocess_image(image)
                outputs = self.model(tensor)
                # No need to apply softmax here since it's already applied in the model's forward method
                return outputs
        except Exception as e:
            logger.error(f"Error getting model predictions: {str(e)}")
            raise

    def detect_animal(self, predictions: Dict[str, torch.Tensor], top_k=5):
        """Detect animal type from model predictions.

        Args:
            predictions: Dictionary of model predictions
            top_k: Number of top predictions to consider

        Returns:
            Tuple of detected animal type and confidence score
        """
        try:
            # Get the animal type predictions
            animal_probs = predictions['animal_type'].squeeze()
            
            # Convert to probabilities if not already
            if not isinstance(animal_probs, torch.Tensor):
                animal_probs = torch.tensor(animal_probs)
            
            # Get the highest probability and its index
            prob, idx = torch.max(animal_probs, dim=0)
            
            # Convert to Python types
            prob = float(prob)
            idx = int(idx)
            
            # Map index to animal type using LABEL_MAPS
            animal_type_map = {v: k for k, v in LABEL_MAPS['animal_type'].items()}
            detected_animal = animal_type_map.get(idx, 'other')
            
            if prob >= self.config.confidence_threshold:
                logger.info(f"Detected {detected_animal} with confidence {prob}")
                return detected_animal, prob
            else:
                logger.info("No animal detected with sufficient confidence")
                return 'other', prob

        except Exception as e:
            logger.error(f"Error in detect_animal: {str(e)}")
            return 'other', 0.0

    def estimate_image_quality(self, image: Image.Image) -> Tuple[str, float]:
        """Estimate image quality based on multiple metrics.

        Args:
            image: Input PIL image

        Returns:
            Tuple of quality label and confidence score
        """
        try:
            gray = image.convert('L')
            img_array = np.array(gray)

            blur = cv2.Laplacian(img_array, cv2.CV_64F).var()
            contrast = np.std(img_array) / 255.0
            median_filtered = cv2.medianBlur(img_array, 3)
            noise = np.mean(np.abs(img_array - median_filtered)) / 255.0

            hist = np.histogram(img_array, bins=256)[0]
            hist_norm = hist / hist.sum()
            mid_tones = np.sum(hist_norm[64:192]) / np.sum(hist_norm)

            quality_score = (
                0.4 * min(blur / 500, 1.0) +
                0.3 * contrast +
                0.2 * (1.0 - noise) +
                0.1 * mid_tones
            )

            if quality_score > 0.45:
                return "good", quality_score
            elif quality_score > 0.25:
                return "medium", quality_score
            else:
                return "poor", quality_score

        except Exception as e:
            logger.error(f"Error estimating image quality: {str(e)}")
            return "poor", 0.0

    def estimate_size(self, image: Image.Image, predictions: Dict[str, torch.Tensor]) -> Tuple[str, float]:
        """Estimate animal size based on model predictions.

        Args:
            image: Input PIL image
            predictions: Dictionary of model predictions

        Returns:
            Tuple of size category and confidence score
        """
        try:
            # Get the size predictions
            size_probs = predictions['size'].squeeze()
            
            # Convert to probabilities if not already
            if not isinstance(size_probs, torch.Tensor):
                size_probs = torch.tensor(size_probs)
            
            # Get the highest probability and its index
            prob, idx = torch.max(size_probs, dim=0)
            
            # Convert to Python types
            prob = float(prob)
            idx = int(idx)
            
            # Map index to size using LABEL_MAPS
            size_map = {v: k for k, v in LABEL_MAPS['size'].items()}
            predicted_size = size_map.get(idx, 'medium')
            
            if prob >= self.config.confidence_threshold:
                logger.info(f"Detected size {predicted_size} with confidence {prob}")
                return predicted_size, prob
            else:
                logger.info("Low confidence in size prediction, defaulting to medium")
                return 'medium', prob

        except Exception as e:
            logger.error(f"Error in size estimation: {str(e)}")
            return 'medium', 0.0

    def detect_context(self, predictions: Dict[str, torch.Tensor], image: Image.Image) -> Tuple[str, float]:
        """Detect context from model predictions.

        Args:
            predictions: Dictionary of model predictions
            image: Input PIL image

        Returns:
            Tuple of context category and confidence score
        """
        try:
            # Get the context predictions
            context_probs = predictions['context'].squeeze()
            
            # Convert to probabilities if not already
            if not isinstance(context_probs, torch.Tensor):
                context_probs = torch.tensor(context_probs)
            
            # Get the highest probability and its index
            prob, idx = torch.max(context_probs, dim=0)
            
            # Convert to Python types
            prob = float(prob)
            idx = int(idx)
            
            # Map index to context using LABEL_MAPS
            context_map = {v: k for k, v in LABEL_MAPS['context'].items()}
            detected_context = context_map.get(idx, 'other')
            
            if prob >= self.config.confidence_threshold:
                logger.info(f"Detected context {detected_context} with confidence {prob}")
                return detected_context, prob
            else:
                logger.info("Low confidence in context prediction, defaulting to home")
                return 'home', prob

        except Exception as e:
            logger.error(f"Error in context detection: {str(e)}")
            return 'home', 0.0

    def analyze_health(self, image: Image.Image) -> dict:
        """Analyze animal health conditions in image.

        Detects visible health issues including:
        - Wounds and injuries
        - Skin conditions
        - Inflammation signs

        Args:
            image: Input PIL image to analyze

        Returns:
            Dict containing:
                - health_status: Detected condition
                - confidence: Detection confidence score
                - details: Additional analysis metrics
        """
        try:
            logger.info("Starting health analysis")
            img_np = np.array(image)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([8, 255, 255])
            lower_red2 = np.array([172, 120, 70])
            upper_red2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            kernel = np.ones((5, 5), np.uint8)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            lbp = local_binary_pattern(gray, 12, 2, method='uniform')
            lbp_threshold = np.mean(lbp) + 2.0 * np.std(lbp)
            lbp_mask = (lbp > lbp_threshold).astype(np.uint8) * 255

            lower_infl = np.array([0, 50, 200])
            upper_infl = np.array([10, 130, 255])
            mask_infl = cv2.inRange(hsv, lower_infl, upper_infl)

            mask_infl = cv2.morphologyEx(mask_infl, cv2.MORPH_OPEN, kernel)
            lbp_mask = cv2.morphologyEx(lbp_mask, cv2.MORPH_OPEN, kernel)

            total_pixels = image.size[0] * image.size[1]
            wounds_pixels = np.sum(mask_red > 0)
            infl_pixels = np.sum(mask_infl > 0)
            skin_pixels = np.sum(lbp_mask > 0)

            wounds_ratio = wounds_pixels / total_pixels
            infl_ratio = infl_pixels / total_pixels
            skin_ratio = skin_pixels / total_pixels

            logger.info(
                f"Analysis ratios - Wounds: {wounds_ratio:.4f}, "
                f"Inflammation: {infl_ratio:.4f}, "
                f"Skin: {skin_ratio:.4f}"
            )

            if wounds_ratio > 0.03 and wounds_ratio < 0.45:
                contours, _ = cv2.findContours(
                    mask_red,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                valid_wound = False
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        valid_wound = True
                        break

                if valid_wound:
                    health_status = "wounds"
                    confidence = min(max(wounds_ratio * 8, 0.6), 1.0)
                    logger.info(f"Detected wounds with ratio {wounds_ratio:.4f}")
                else:
                    health_status = "none"
                    confidence = 0.7
                    logger.info("Detected red areas but no wound patterns")
            elif skin_ratio > 0.15 and skin_ratio < 0.55:
                health_status = "skin_issues"
                confidence = min(max(skin_ratio * 7, 0.6), 1.0)
                logger.info(f"Detected skin issues with ratio {skin_ratio:.4f}")
            elif infl_ratio > 0.05 and infl_ratio < 0.35:
                health_status = "other"
                confidence = min(max(infl_ratio * 8, 0.6), 1.0)
                logger.info(
                    f"Detected inflammation with ratio {infl_ratio:.4f}"
                )
            else:
                health_status = "none"
                confidence = 0.7
                logger.info("No health issues detected")

            result = {
                "health_status": health_status,
                "confidence": float(confidence),
                "details": {
                    "wounds_ratio": float(wounds_ratio),
                    "skin_issues_ratio": float(skin_ratio),
                    "inflammation_ratio": float(infl_ratio)
                }
            }

            logger.info(f"Final health analysis result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in health analysis: {str(e)}")
            return {
                "health_status": "none",
                "confidence": 0.0,
                "details": {
                    "error": str(e)
                }
            }

    def analyze_pregnancy(self, image, mask):
        """Analyze pregnancy indicators in animal image.

        Uses multiple indicators including:
        - Abdominal measurements
        - Body symmetry
        - Shape analysis
        - Mammary development

        Args:
            image: Input numpy array image
            mask: Animal segmentation mask

        Returns:
            Tuple containing:
                - Pregnancy state ('none', 'possible', 'visible')
                - Confidence score for the detection
        """
        try:
            logger.info("Starting pregnancy analysis")

            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            mask = mask.astype(np.float32)

            height, width = mask.shape[:2]
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return "none", 0.0

            y1, y2 = y_indices.min(), y_indices.max()
            x1, x2 = x_indices.min(), x_indices.max()

            body_height = y2 - y1
            body_width = x2 - x1

            abdomen_y1 = y1 + int(body_height * 0.3)
            abdomen_y2 = y1 + int(body_height * 0.7)

            abdomen_mask = mask[abdomen_y1:abdomen_y2, x1:x2]
            abdomen_region = image[abdomen_y1:abdomen_y2, x1:x2].copy()

            if abdomen_region.size == 0:
                return "none", 0.0

            abdomen_width = np.sum(abdomen_mask, axis=1)
            max_abdomen_width = abdomen_width.max()

            shoulder_width = np.sum(mask[y1:abdomen_y1, x1:x2], axis=1).max()
            hip_width = np.sum(mask[abdomen_y2:y2, x1:x2], axis=1).max()

            abdomen_profile = abdomen_width / max_abdomen_width
            profile_symmetry = np.abs(
                abdomen_profile - np.flip(abdomen_profile)
            ).mean()

            abdomen_contour = (abdomen_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                abdomen_contour,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)

                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0

                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                convexity = area / hull_area if hull_area > 0 else 0

                expected_area = (
                    abdomen_region.shape[0] * abdomen_region.shape[1] * 0.5
                )
                area_ratio = area / expected_area
            else:
                circularity = convexity = area_ratio = 0

            max_width_ratio = max_abdomen_width / max(shoulder_width, hip_width)
            pregnancy_indicators = {
                'abdomen_width': max_width_ratio > 1.2,
                'symmetry': profile_symmetry < 0.2,
                'shape': circularity > 0.6 and convexity > 0.85,
                'area': area_ratio > 0.4,
                'profile': np.std(abdomen_profile) < 0.3
            }

            if y2 + int(body_height * 0.1) < height:
                mammary_y2 = y2 + int(body_height * 0.1)
                mammary_region = mask[y2:mammary_y2, x1:x2]
                mammary_visibility = np.sum(mammary_region) / mammary_region.size
                pregnancy_indicators['mammary_development'] = (
                    mammary_visibility > 0.3
                )

            positive_indicators = sum(
                1 for v in pregnancy_indicators.values() if v
            )
            pregnancy_score = positive_indicators / len(pregnancy_indicators)

            if pregnancy_score > 0.7:
                pregnancy_state = "visible"
                confidence = min(pregnancy_score * 1.2, 1.0)
            elif pregnancy_score > 0.5:
                pregnancy_state = "possible"
                confidence = pregnancy_score
            else:
                pregnancy_state = "none"
                confidence = 1.0 - pregnancy_score

            logger.info(
                f"Pregnancy Analysis - State: {pregnancy_state}, "
                f"Score: {pregnancy_score:.3f}"
            )
            logger.info(f"Pregnancy Indicators: {pregnancy_indicators}")

            return pregnancy_state, float(confidence)

        except Exception as e:
            logger.error(f"Error in pregnancy analysis: {str(e)}", exc_info=True)
            return "none", 0.0

    def analyze_body_condition(self, image: Image.Image) -> Tuple[str, float]:
        """Analyze body condition focusing on malnutrition and overweight.

        Args:
            image: Input PIL image

        Returns:
            Tuple of body condition category and confidence score
        """
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            height, width = edges.shape
            regions = np.array_split(edges, 3, axis=1)
            middle_region = regions[1]
            edge_density = np.sum(middle_region > 0) / middle_region.size

            if edge_density > 0.1:
                logger.info(f"High edge density detected: {edge_density:.3f}")
                return 'underweight', 0.9

            try:
                _, binary = cv2.threshold(
                    gray, 127, 255, cv2.THRESH_BINARY
                )
                contours, _ = cv2.findContours(
                    binary,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = float(w) / h
                    if perimeter > 0:
                        roundness = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        roundness = 0

                    relative_area = area / (height * width)

                    if (roundness > 0.75 and
                            aspect_ratio > 0.85 and
                            relative_area > 0.4):
                        return 'overweight', min(0.8, roundness)

                return 'normal', 0.7

            except Exception as e:
                logger.error(f"Error in body condition analysis: {str(e)}")
                return 'normal', 0.6

        except Exception as e:
            logger.error(f"Error in body condition analysis: {str(e)}")
            return 'normal', 0.5

    def get_animal_segmentation(self, image):
        """Get animal segmentation mask and bounding box.

        Args:
            image: Input PIL image

        Returns:
            Dictionary containing mask, bounding box and confidence
        """
        try:
            results = self.yolo_model(image, stream=True)

            for result in results:
                if result.masks is not None:
                    masks = result.masks.data
                    boxes = result.boxes.data

                    areas = [(mask.sum(), i) for i, mask in enumerate(masks)]
                    if areas:
                        _, max_idx = max(areas)
                        animal_mask = masks[max_idx]
                        animal_box = boxes[max_idx]

                        return {
                            'mask': animal_mask,
                            'box': animal_box,
                            'confidence': float(result.boxes.conf[max_idx])
                        }

            return None
        except Exception as e:
            logger.error(f"Error in animal segmentation: {str(e)}")
            return None

    def __call__(self, image: Image.Image) -> Dict[str, Any]:
        """Perform complete image analysis.

        Executes full analysis pipeline including:
        - Animal detection and classification
        - Health condition assessment
        - Pregnancy indicator analysis
        - Body condition evaluation
        - Image quality check

        Args:
            image: Input PIL image to analyze

        Returns:
            Dict containing analysis results and confidence scores
            for all evaluated aspects
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            predictions = self.predict(image)
            animal_type, confidence = self.detect_animal(predictions)

            if confidence < self.config.discard_threshold:
                return {
                    "should_discard": True,
                    "reason": (
                        f"Low confidence detection ({confidence:.3f})"
                    ),
                    "confidence": float(confidence)
                }

            if animal_type == 'other':
                return {
                    "should_discard": True,
                    "reason": "No animal detected",
                    "confidence": float(confidence)
                }

            segmentation = self.get_animal_segmentation(image)
            if segmentation is None:
                logger.warning("No segmentation mask found for analysis")
                return {
                    "should_discard": True,
                    "reason": "No clear animal segmentation",
                    "confidence": float(confidence)
                }

            size, size_conf = self.estimate_size(image, predictions)
            context, context_conf = self.detect_context(predictions, image)
            quality, quality_conf = self.estimate_image_quality(image)

            img_array = np.array(image)

            health_result = self.analyze_health(image)
            health_state = health_result.get('health_status', 'unknown')
            health_conf = health_result.get('confidence', 0.0)
            logger.info(
                f"Health analysis: state={health_state}, conf={health_conf}"
            )

            pregnancy_state, preg_conf = self.analyze_pregnancy(
                img_array,
                segmentation['mask']
            )
            logger.info(
                f"Pregnancy analysis: state={pregnancy_state}, "
                f"conf={preg_conf}"
            )

            condition, cond_conf = self.analyze_body_condition(image)

            logger.info(
                f"Confidence values - Animal: {confidence:.3f}, "
                f"Size: {size_conf:.3f}, Health: {health_conf:.3f}, "
                f"Pregnancy: {preg_conf:.3f}, Body: {cond_conf:.3f}, "
                f"Quality: {quality_conf:.3f}, Context: {context_conf:.3f}"
            )

            health_menu_state = self.config.menu.health_issues.get(
                health_state,
                "none"
            )
            pregnancy_menu_state = self.config.menu.pregnancy.get(
                pregnancy_state,
                "none"
            )

            base_labels = {
                "animal_type": animal_type,
                "size": size,
                "body_condition": (
                    self.config.menu.body_condition[condition]
                ),
                "visible_health_issues": health_menu_state,
                "pregnancy_indicators": pregnancy_menu_state,
                "image_quality": quality,
                "context": context
            }

            return {
                "should_discard": False,
                "labels": base_labels,
                "confidence": float(confidence),
                "size_confidence": float(size_conf),
                "health_confidence": float(health_conf),
                "pregnancy_confidence": float(preg_conf),
                "body_condition_confidence": float(cond_conf),
                "quality_confidence": float(quality_conf),
                "context_confidence": float(context_conf)
            }

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise ValueError(f"Error processing image: {str(e)}")
