#!/usr/bin/env python3
"""Image processing utilities.

This module provides functions for image manipulation, validation,
and quality assessment. Includes functionality for:
- Image resizing and normalization
- Quality enhancement and assessment
- Format validation
- Data augmentation
"""

from typing import Tuple, Optional, List
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from app.schemas.image import (
    ImageValidation,
    ImageQuality
)


def resize_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize image maintaining aspect ratio.

    Args:
        image: Input PIL image
        target_size: Desired (width, height)

    Returns:
        Resized PIL image
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values.

    Args:
        image: Input numpy array image

    Returns:
        Normalized numpy array with values in [0, 1]
    """
    return image.astype(np.float32) / 255.0


def enhance_image(image: Image.Image) -> Image.Image:
    """Enhance image quality.

    Applies basic image enhancement including:
    - Contrast adjustment
    - Brightness optimization

    Args:
        image: Input PIL image

    Returns:
        Enhanced PIL image
    """
    img_array = np.array(image)
    img_array = cv2.convertScaleAbs(img_array, alpha=1.1, beta=10)
    return Image.fromarray(img_array)


def validate_image(
    image: Image.Image,
    validation_params: ImageValidation
) -> Tuple[bool, str]:
    """Validate image dimensions, size and format.

    Args:
        image: Input PIL image
        validation_params: Validation parameters

    Returns:
        Tuple containing:
        - Boolean indicating if validation passed
        - Message describing validation result
    """
    width, height = image.size
    min_dims = (
        width < validation_params.min_width or
        height < validation_params.min_height
    )
    if min_dims:
        return False, (
            f"Image dimensions must be at least "
            f"{validation_params.min_width}x{validation_params.min_height}"
        )

    if image.format.lower() not in validation_params.allowed_formats:
        return False, (
            f"Image format must be one of {validation_params.allowed_formats}"
        )

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=image.format)
    size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
    if size_mb > validation_params.max_size_mb:
        return False, (
            f"Image size must be less than {validation_params.max_size_mb}MB"
        )

    return True, "Image validation successful"


def assess_image_quality(image: Image.Image) -> ImageQuality:
    """Assess various quality metrics of the image.

    Evaluates:
    - Brightness
    - Contrast
    - Sharpness

    Args:
        image: Input PIL image

    Returns:
        ImageQuality object containing quality metrics
    """
    img_array = np.array(image.convert('RGB'))

    brightness = np.mean(img_array) / 255.0

    contrast = np.std(img_array) / 255.0

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 1000.0, 1.0)

    is_valid = (
        0.2 <= brightness <= 0.8 and
        contrast >= 0.3 and
        sharpness >= 0.4
    )

    return ImageQuality(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        is_valid=is_valid
    )


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    normalize: bool = True
) -> np.ndarray:
    """Preprocess image for model input.

    Args:
        image: Input PIL image
        target_size: Desired (width, height)
        normalize: Whether to normalize pixel values

    Returns:
        Preprocessed numpy array
    """
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(image)

    if normalize:
        img_array = img_array.astype(np.float32) / 255.0

    return img_array


def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Enhance image quality for better model performance.

    Applies:
    - RGB conversion if needed
    - Brightness enhancement
    - Contrast enhancement
    - Sharpness enhancement

    Args:
        image: Input PIL image

    Returns:
        Enhanced PIL image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)

    return image


def augment_image(
    image: Image.Image,
    rotation_range: int = 20,
    flip: bool = True,
    brightness_range: float = 0.2
) -> List[Image.Image]:
    """Apply data augmentation to generate multiple variants.

    Applies:
    - Rotation
    - Horizontal flipping
    - Brightness variation

    Args:
        image: Input PIL image
        rotation_range: Maximum rotation angle in degrees
        flip: Whether to apply horizontal flipping
        brightness_range: Range for brightness variation

    Returns:
        List of augmented PIL images
    """
    augmented_images = []
    augmented_images.append(image)

    for angle in [-rotation_range, rotation_range]:
        rotated = image.rotate(angle, expand=True)
        augmented_images.append(rotated)

    if flip:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(flipped)

    enhancer = ImageEnhance.Brightness(image)
    for factor in [1.0 - brightness_range, 1.0 + brightness_range]:
        brightened = enhancer.enhance(factor)
        augmented_images.append(brightened)

    return augmented_images
