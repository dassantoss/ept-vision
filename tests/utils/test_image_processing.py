import pytest
from PIL import Image
import numpy as np

from app.utils.image_processing import (
    validate_image,
    assess_image_quality,
    preprocess_image,
    enhance_image_quality,
    augment_image
)
from app.schemas.image import ImageValidation

@pytest.fixture
def test_image():
    # Create a test image with a gradient
    width, height = 224, 224
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        array[:, i] = [i % 256, (i + 85) % 256, (i + 170) % 256]
    return Image.fromarray(array)

def test_validate_image(test_image):
    validation_params = ImageValidation()
    test_image.format = 'JPEG'  # Set format manually for test
    
    # Test valid image
    is_valid, message = validate_image(test_image, validation_params)
    assert is_valid
    assert "successful" in message
    
    # Test invalid dimensions
    small_image = test_image.resize((100, 100))
    small_image.format = 'JPEG'
    is_valid, message = validate_image(small_image, validation_params)
    assert not is_valid
    assert "dimensions" in message

def test_assess_image_quality(test_image):
    quality = assess_image_quality(test_image)
    assert 0 <= quality.brightness <= 1
    assert 0 <= quality.contrast <= 1
    assert 0 <= quality.sharpness <= 1
    assert isinstance(quality.is_valid, bool)

def test_preprocess_image(test_image):
    target_size = (224, 224)
    processed = preprocess_image(test_image, target_size)
    assert processed.shape == (*target_size, 3)
    assert processed.dtype == np.float32
    assert 0 <= processed.min() <= processed.max() <= 1

def test_enhance_image_quality(test_image):
    enhanced = enhance_image_quality(test_image)
    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == test_image.size
    assert enhanced.mode == 'RGB'

def test_augment_image(test_image):
    augmented = augment_image(test_image)
    assert len(augmented) > 1
    for img in augmented:
        assert isinstance(img, Image.Image) 