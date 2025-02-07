#!/usr/bin/env python3
"""
Pet Analysis Model Module.

This module implements a unified model for comprehensive pet image analysis,
combining multiple specialized models for different aspects of pet assessment.
The model provides detailed analysis of pet images including animal type
detection, health assessment, and environmental context evaluation.

Classes:
    MultiTaskModel: Neural network architecture for multi-task analysis
    PetAnalysisModel: Main class for pet image analysis
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor
)
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Label mappings
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
    """Menu option mappings for classification results."""
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

class MultiTaskModel(nn.Module):
    """Multi-task model architecture for comprehensive pet analysis."""
    
    def __init__(self):
        """Initialize model architecture."""
        super().__init__()
        
        # Load EfficientNet backbone
        self.backbone = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_efficientnet_b0',
            pretrained=True
        )
        
        # Remove original classifier
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
        """Forward pass through the model."""
        features = self.features(x)
        features = features.mean([2, 3])  # Global average pooling
        
        return {
            'animal_type': F.softmax(self.animal_type(features), dim=1),
            'size': F.softmax(self.size(features), dim=1),
            'body_condition': F.softmax(self.body_condition(features), dim=1),
            'health_issues': F.softmax(self.health_issues(features), dim=1),
            'pregnancy': F.softmax(self.pregnancy(features), dim=1),
            'quality': F.softmax(self.image_quality(features), dim=1),
            'context': F.softmax(self.context(features), dim=1)
        }

class PetAnalysisModel:
    """Model for comprehensive pet image analysis."""
    
    def __init__(self):
        """Initialize model components and load required weights."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.initialize_model()

        # Initialize additional models
        model_path = 'app/models/weights/yolov8x-seg.pt'
        self.yolo_model = YOLO(model_path)

        model_name = "microsoft/resnet-50"
        self.gender_classifier = AutoModelForImageClassification.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.vet_model = models.resnet50(pretrained=True)
        
        # Move models to device
        self.gender_classifier.to(self.device)
        self.vet_model.to(self.device)

        # Configuration parameters
        self.confidence_threshold = 0.15
        self.size_thresholds = {
            'small': 0.25,
            'medium': 0.50,
            'large': 1.0
        }
        self.default_context = 'home'
        self.menu = MenuMappings()

        logger.info("Models loaded successfully")

    def initialize_model(self):
        """Initialize the model and load trained weights."""
        try:
            self.model = MultiTaskModel().to(self.device)
            weights_path = Path(__file__).parent / "weights" / "advanced" / "best_model.pth"
            
            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.eval()
                logger.info(f"Loaded custom weights from {weights_path}")
            else:
                raise FileNotFoundError(f"Weights not found at {weights_path}")
                
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Get model predictions for an image."""
        try:
            with torch.no_grad():
                tensor = self.preprocess_image(image)
                outputs = self.model(tensor)
                return outputs
        except Exception as e:
            logger.error(f"Error getting model predictions: {str(e)}")
            raise

    def detect_animal(self, predictions: Dict[str, torch.Tensor], top_k=5):
        """Detect animal type from model predictions."""
        try:
            animal_probs = predictions['animal_type'].squeeze()
            
            if not isinstance(animal_probs, torch.Tensor):
                animal_probs = torch.tensor(animal_probs)
            
            prob, idx = torch.max(animal_probs, dim=0)
            prob = float(prob)
            idx = int(idx)
            
            animal_type_map = {v: k for k, v in LABEL_MAPS['animal_type'].items()}
            detected_animal = animal_type_map.get(idx, 'other')
            
            if prob >= self.confidence_threshold:
                logger.info(f"Detected {detected_animal} with confidence {prob}")
                return detected_animal, prob
            else:
                logger.info("No animal detected with sufficient confidence")
                return 'other', prob

        except Exception as e:
            logger.error(f"Error in detect_animal: {str(e)}")
            return 'other', 0.0

    def estimate_image_quality(self, image: Image.Image) -> Tuple[str, float]:
        """Estimate image quality based on multiple metrics."""
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
        """Estimate animal size based on model predictions."""
        try:
            size_probs = predictions['size'].squeeze()
            
            if not isinstance(size_probs, torch.Tensor):
                size_probs = torch.tensor(size_probs)
            
            prob, idx = torch.max(size_probs, dim=0)
            prob = float(prob)
            idx = int(idx)
            
            size_map = {v: k for k, v in LABEL_MAPS['size'].items()}
            predicted_size = size_map.get(idx, 'medium')
            
            if prob >= self.confidence_threshold:
                logger.info(f"Detected size {predicted_size} with confidence {prob}")
                return predicted_size, prob
            else:
                logger.info("Low confidence in size prediction, defaulting to medium")
                return 'medium', prob

        except Exception as e:
            logger.error(f"Error in size estimation: {str(e)}")
            return 'medium', 0.0

    def detect_context(self, predictions: Dict[str, torch.Tensor], image: Image.Image) -> Tuple[str, float]:
        """Detect context from model predictions."""
        try:
            context_probs = predictions['context'].squeeze()
            
            if not isinstance(context_probs, torch.Tensor):
                context_probs = torch.tensor(context_probs)
            
            prob, idx = torch.max(context_probs, dim=0)
            prob = float(prob)
            idx = int(idx)
            
            context_map = {v: k for k, v in LABEL_MAPS['context'].items()}
            detected_context = context_map.get(idx, 'other')
            
            if prob >= self.confidence_threshold:
                logger.info(f"Detected context {detected_context} with confidence {prob}")
                return detected_context, prob
            else:
                logger.info("Low confidence in context prediction, defaulting to home")
                return 'home', prob

        except Exception as e:
            logger.error(f"Error in context detection: {str(e)}")
            return 'home', 0.0

    def analyze_health(self, image: Image.Image) -> dict:
        """Analyze animal health conditions in image."""
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
                logger.info(f"Detected inflammation with ratio {infl_ratio:.4f}")
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
        """Analyze pregnancy indicators in animal image."""
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
        """Analyze body condition focusing on malnutrition and overweight."""
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
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
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
                    roundness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    relative_area = area / (height * width)

                    if roundness > 0.75 and aspect_ratio > 0.85 and relative_area > 0.4:
                        return 'overweight', min(0.8, roundness)

                return 'normal', 0.7

            except Exception as e:
                logger.error(f"Error in body condition analysis: {str(e)}")
                return 'normal', 0.6

        except Exception as e:
            logger.error(f"Error in body condition analysis: {str(e)}")
            return 'normal', 0.5

    def get_animal_segmentation(self, image):
        """Get animal segmentation mask and bounding box."""
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
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Perform complete image analysis."""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            predictions = self.predict(image)
            animal_type, confidence = self.detect_animal(predictions)

            if confidence < self.confidence_threshold:
                return {
                    "should_discard": True,
                    "reason": f"Low confidence detection ({confidence:.3f})",
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

            pregnancy_state, preg_conf = self.analyze_pregnancy(
                img_array,
                segmentation['mask']
            )

            condition, cond_conf = self.analyze_body_condition(image)

            logger.info(
                f"Confidence values - Animal: {confidence:.3f}, "
                f"Size: {size_conf:.3f}, Health: {health_conf:.3f}, "
                f"Pregnancy: {preg_conf:.3f}, Body: {cond_conf:.3f}, "
                f"Quality: {quality_conf:.3f}, Context: {context_conf:.3f}"
            )

            health_menu_state = self.menu.health_issues.get(health_state, "none")
            pregnancy_menu_state = self.menu.pregnancy.get(pregnancy_state, "none")

            base_labels = {
                "animal_type": animal_type,
                "size": size,
                "body_condition": self.menu.body_condition[condition],
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

# Create singleton instance
model = PetAnalysisModel() 
