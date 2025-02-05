#!/usr/bin/env python3
"""Model utility functions for PyTorch models.

This module provides common utility functions for working with PyTorch models:
- Model input/output processing
- Weight loading
- Device management
- Transformation pipelines
"""

from typing import Dict, Any, Tuple
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from app.core.model_config import ModelConfig


def get_transform(config: ModelConfig) -> transforms.Compose:
    """Get the transformation pipeline for a model.

    Args:
        config: Model configuration object

    Returns:
        Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image: Image.Image, config: ModelConfig) -> torch.Tensor:
    """Preprocess an image for model input.

    Args:
        image: Input PIL image
        config: Model configuration object

    Returns:
        Preprocessed tensor ready for model input
    """
    transform = get_transform(config)
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def postprocess_predictions(
    predictions: torch.Tensor,
    config: ModelConfig
) -> Dict[str, Any]:
    """Convert model predictions to human-readable format.

    Args:
        predictions: Raw model output tensor
        config: Model configuration object

    Returns:
        Dictionary containing:
        - predicted_class: Main prediction
        - confidence: Prediction confidence
        - top_predictions: Top 3 predictions with probabilities
    """
    # Get probabilities
    probs = torch.nn.functional.softmax(predictions, dim=1)

    # Get top prediction
    prob, idx = torch.max(probs, dim=1)

    # Convert to Python types
    predicted_class = config.class_names[idx.item()]
    confidence = prob.item()

    # Get top-3 predictions with probabilities
    top3_probs, top3_idx = torch.topk(
        probs,
        min(3, config.num_classes),
        dim=1
    )
    top3_predictions = [
        {
            "class": config.class_names[idx.item()],
            "probability": prob.item()
        }
        for prob, idx in zip(top3_probs[0], top3_idx[0])
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_predictions": top3_predictions
    }


def load_model_weights(model: torch.nn.Module, weights_path: str) -> None:
    """Load model weights from file.

    Args:
        model: PyTorch model instance
        weights_path: Path to weights file

    Raises:
        RuntimeError: If loading weights fails
    """
    try:
        state_dict = torch.load(
            weights_path,
            map_location=torch.device('cpu')
        )
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {str(e)}")


def get_device() -> torch.device:
    """Get the appropriate device for model inference.

    Returns:
        torch.device: CUDA if available, CPU otherwise
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
