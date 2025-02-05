#!/usr/bin/env python3
"""
Model configuration module.

This module defines configuration settings for all machine learning models
used in the application, including input specifications, class definitions,
and model architecture details.
"""

from typing import Dict, Any, Tuple, List
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """
    Configuration settings for machine learning models.

    This class defines the structure and validation for model configurations,
    ensuring all required parameters are properly specified.

    Attributes:
        name: Unique identifier for the model
        version: Model version string
        input_size: Tuple of (height, width) for input images
        num_classes: Number of output classes
        class_names: List of class names corresponding to outputs
        model_type: Architecture type (e.g., 'efficientnet_b0', 'vit_base')
        weights_path: Path to the model weights file
    """
    name: str
    version: str
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int
    class_names: List[str]
    model_type: str
    weights_path: str


# Disease detection model configuration
DISEASE_DETECTION_CONFIG = ModelConfig(
    name="disease_detection",
    version="1.0.0",
    input_size=(224, 224),
    num_classes=4,  # External visible conditions
    class_names=[
        "healthy",
        "skin_infection",
        "eye_infection",
        "external_wounds"
    ],
    model_type="efficientnet_b0",
    weights_path="app/models/disease_detection/weights/model.pt"
)

# Nutrition analysis model configuration
NUTRITION_ANALYSIS_CONFIG = ModelConfig(
    name="nutrition_analysis",
    version="1.0.0",
    input_size=(224, 224),
    num_classes=4,  # Nutritional states
    class_names=[
        "optimal",
        "underweight",
        "overweight",
        "malnutrition"
    ],
    model_type="vit_base",
    weights_path="app/models/nutrition_analysis/weights/model.pt"
)

# Pregnancy detection model configuration
PREGNANCY_DETECTION_CONFIG = ModelConfig(
    name="pregnancy_detection",
    version="1.0.0",
    input_size=(224, 224),
    num_classes=2,  # Binary: pregnant or not
    class_names=[
        "not_pregnant",
        "pregnant"
    ],
    model_type="efficientnet_b0",
    weights_path="app/models/pregnancy_detection/weights/model.pt"
)

# Pet recognition model configuration
PET_RECOGNITION_CONFIG = ModelConfig(
    name="pet_recognition",
    version="1.0.0",
    input_size=(224, 224),
    num_classes=10,  # Common breeds
    class_names=[
        "labrador",
        "german_shepherd",
        "golden_retriever",
        "bulldog",
        "poodle",
        "beagle",
        "rottweiler",
        "yorkshire",
        "boxer",
        "other"
    ],
    model_type="vit_base",
    weights_path="app/models/pet_recognition/weights/model.pt"
)

# Dictionary mapping model names to their configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "disease_detection": DISEASE_DETECTION_CONFIG,
    "nutrition_analysis": NUTRITION_ANALYSIS_CONFIG,
    "pregnancy_detection": PREGNANCY_DETECTION_CONFIG,
    "pet_recognition": PET_RECOGNITION_CONFIG
}
