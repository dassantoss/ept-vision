#!/usr/bin/env python3
"""
Pet detection model implementation.

This module implements a YOLOv8-based model for detecting and analyzing pets
in images. It includes functionality for pet detection, breed classification,
size estimation, and image quality assessment.
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
import os

from app.core.logging import get_logger

logger = get_logger("ept_vision.pet_detection")


class PetDetectionModel:
    """
    Pet detection model implementation using YOLOv8.

    This class implements a YOLOv8-based model for detecting pets in images,
    with additional functionality for breed classification, size estimation,
    and image quality assessment.

    Attributes:
        model: YOLOv8 model instance
        CAT_CLASS: COCO dataset index for cats
        DOG_CLASS: COCO dataset index for dogs
        CONFIDENCE_THRESHOLD: Detection confidence threshold
        DOG_SIZES: Mapping of dog breeds to their typical sizes
    """

    def __init__(self):
        """Initialize the model with YOLOv8x for best accuracy."""
        try:
            # Use YOLOv8x for maximum precision
            self.model = YOLO('app/models/pet_detection/weights/yolov8x.pt')

            # Classes of interest (COCO dataset)
            self.CAT_CLASS = 15  # Cat index in COCO
            self.DOG_CLASS = 16  # Dog index in COCO

            # Adjusted confidence threshold
            # Lowered from 0.45 for better detection of small animals
            self.CONFIDENCE_THRESHOLD = 0.35

            # Mapping of dog breeds by typical size
            self.DOG_SIZES = {
                'small': [
                    'chihuahua', 'pomeranian', 'yorkshire', 'maltese',
                    'shih-tzu', 'toy poodle', 'miniature poodle',
                    'dachshund', 'terrier'
                ],
                'medium': [
                    'beagle', 'bulldog', 'boxer', 'collie', 'dalmatian',
                    'standard poodle', 'labrador', 'golden retriever'
                ],
                'large': [
                    'german shepherd', 'rottweiler', 'great dane',
                    'saint bernard', 'mastiff', 'newfoundland', 'husky',
                    'malamute'
                ]
            }

            logger.info("Pet detection model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def estimate_size_from_breed(self, breed_name: str) -> Optional[str]:
        """
        Estimate animal size based on breed name.

        Args:
            breed_name: Name of the detected breed

        Returns:
            Optional[str]: Size category ('small', 'medium', 'large')
                         or None if size cannot be determined
        """
        breed_lower = breed_name.lower()
        logger.info(f"Estimating size for breed: {breed_name}")

        for size, breeds in self.DOG_SIZES.items():
            if any(breed_keyword in breed_lower for breed_keyword in breeds):
                logger.info(f"Matched breed {breed_name} to size {size}")
                return size

        logger.info(f"Could not determine size from breed {breed_name}")
        return None

    def analyze_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze image quality based on contrast and brightness.

        Args:
            image: Input PIL Image

        Returns:
            Dict[str, Any]: Quality analysis results with contrast,
                          brightness, and quality assessment
        """
        try:
            gray_img = np.array(image.convert('L'))
            contrast = np.std(gray_img) / 255.0
            brightness = np.mean(gray_img) / 255.0

            quality = (
                "good"
                if (contrast > 0.15 and 0.2 < brightness < 0.8)
                else "poor"
            )

            analysis = {
                "contrast": float(contrast),
                "brightness": float(brightness),
                "quality": quality
            }

            logger.info(
                f"Image analysis - Contrast: {contrast:.3f}, "
                f"Brightness: {brightness:.3f}, Quality: {quality}"
            )
            return analysis

        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return {"contrast": 0.0, "brightness": 0.0, "quality": "poor"}

    def __call__(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect and analyze pets in an image.

        Args:
            image: Input PIL Image

        Returns:
            Dict[str, Any]: Detection results with pet type, size,
                           confidence scores, and quality assessment

        Raises:
            ValueError: If image processing fails
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_np = np.array(image)
            results = self.model(image_np, conf=self.CONFIDENCE_THRESHOLD)[0]

            logger.info("Processing raw detections")
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                logger.info(
                    f"Class: {cls} (CAT={self.CAT_CLASS}, "
                    f"DOG={self.DOG_CLASS}), Confidence: {conf:.3f}"
                )

            pet_detections = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in [self.CAT_CLASS, self.DOG_CLASS]:
                    bbox = box.xyxy[0].tolist()
                    pet_detections.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': bbox
                    })

                    logger.info(
                        f"Valid pet detection - "
                        f"Type: {'Cat' if cls == self.CAT_CLASS else 'Dog'}, "
                        f"Confidence: {conf:.3f}, BBox: {bbox}"
                    )
            
            if not pet_detections:
                logger.info("No valid pet detections found in image")
                return {
                    "should_discard": True,
                    "reason": "No pets detected",
                    "confidence": 0.0
                }

            best_detection = max(pet_detections, key=lambda x: x['confidence'])
            logger.info(
                f"Best detection - "
                f"Type: {'Cat' if best_detection['class'] == self.CAT_CLASS else 'Dog'}, "
                f"Confidence: {best_detection['confidence']:.3f}"
            )

            animal_type = (
                'cat' if best_detection['class'] == self.CAT_CLASS else 'dog'
            )

            size = None
            if animal_type == 'dog':
                breed_results = self.model.predict(
                    image_np,
                    classes=[self.DOG_CLASS],
                    conf=0.3
                )[0]
                if len(breed_results.boxes) > 0:
                    breed_name = breed_results.names[
                        int(breed_results.boxes[0].cls[0])
                    ]
                    logger.info(f"Detected dog breed: {breed_name}")
                    size = self.estimate_size_from_breed(breed_name)

            if size is None:
                x1, y1, x2, y2 = best_detection['bbox']
                bbox_area = (x2 - x1) * (y2 - y1)
                image_area = image.size[0] * image.size[1]
                relative_area = bbox_area / image_area

                if relative_area < 0.1:
                    size = 'small'
                elif relative_area < 0.3:
                    size = 'medium'
                else:
                    size = 'large'

                logger.info(
                    f"Size determined by relative area: {relative_area:.3f}, "
                    f"Assigned size: {size}"
                )

            quality_analysis = self.analyze_image_quality(image)

            result = {
                "should_discard": False,
                "labels": {
                    "animal_type": animal_type,
                    "size": size,
                    "body_condition": "normal",
                    "visible_health_issues": "none",
                    "pregnancy_indicators": "none",
                    "image_quality": quality_analysis["quality"],
                    "context": "home"
                },
                "confidence": float(best_detection['confidence']),
                "num_animals": len(pet_detections),
                "all_detections": pet_detections,
                "quality_analysis": quality_analysis
            }

            logger.info(f"Final result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise ValueError(f"Error processing image: {str(e)}")
