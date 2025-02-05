import pytest
import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
from app.tools.dataset_manager import DatasetManager

@pytest.fixture
def test_image():
    """Create a test image."""
    width, height = 100, 100
    image = Image.new('RGB', (width, height), color='white')
    return image

@pytest.fixture
def dataset_manager(tmp_path):
    """Create a dataset manager with temporary directory."""
    os.environ["DATA_DIR"] = str(tmp_path)
    return DatasetManager()

def test_discard_image(dataset_manager, test_image):
    """Test discarding an image."""
    # Save a test image
    image_path = os.path.join(dataset_manager.raw_dir, "test_image.jpg")
    test_image.save(image_path)
    
    # Discard the image
    success = dataset_manager.discard_image("test_image.jpg")
    assert success is True
    
    # Verify discarded images file
    with open(dataset_manager.discarded_file, 'r') as f:
        discarded_data = json.load(f)
    
    assert "test_image.jpg" in discarded_data["discarded_images"]
    assert "timestamp" in discarded_data["discarded_images"]["test_image.jpg"]
    assert "reason" in discarded_data["discarded_images"]["test_image.jpg"]

def test_is_image_discarded(dataset_manager, test_image):
    """Test checking if an image is discarded."""
    # Save and discard a test image
    image_path = os.path.join(dataset_manager.raw_dir, "test_image.jpg")
    test_image.save(image_path)
    dataset_manager.discard_image("test_image.jpg")
    
    # Check if image is discarded
    assert dataset_manager.is_image_discarded("test_image.jpg") is True
    assert dataset_manager.is_image_discarded("nonexistent.jpg") is False

def test_get_unlabeled_images_excludes_discarded(dataset_manager, test_image):
    """Test that get_unlabeled_images excludes discarded images."""
    # Save multiple test images
    for i in range(3):
        image_path = os.path.join(dataset_manager.raw_dir, f"test_image_{i}.jpg")
        test_image.save(image_path)
    
    # Discard one image
    dataset_manager.discard_image("test_image_1.jpg")
    
    # Get unlabeled images
    unlabeled = dataset_manager.get_unlabeled_images()
    assert "test_image_1.jpg" not in unlabeled
    assert len(unlabeled) == 2
    assert "test_image_0.jpg" in unlabeled
    assert "test_image_2.jpg" in unlabeled

def test_dataset_stats_with_discarded(dataset_manager, test_image):
    """Test that dataset statistics handle discarded images correctly."""
    # Save multiple test images
    for i in range(4):
        image_path = os.path.join(dataset_manager.raw_dir, f"test_image_{i}.jpg")
        test_image.save(image_path)
    
    # Discard one image
    dataset_manager.discard_image("test_image_1.jpg")
    
    # Label one image
    dataset_manager.update_labels("test_image_2.jpg", {
        "animal_type": "dog",
        "size": "medium"
    })
    
    # Get statistics
    stats = dataset_manager.get_dataset_stats()
    
    assert stats["total_images"] == 3  # Excluding discarded
    assert stats["labeled_images"] == 1
    assert stats["unlabeled_images"] == 2
    assert stats["discarded_images"] == 1
    assert stats["distributions"]["animal_type"]["dog"] == 1

def test_discard_labeled_image(dataset_manager, test_image):
    """Test discarding an image that was previously labeled."""
    # Save and label a test image
    image_path = os.path.join(dataset_manager.raw_dir, "test_image.jpg")
    test_image.save(image_path)
    
    dataset_manager.update_labels("test_image.jpg", {
        "animal_type": "dog",
        "size": "medium"
    })
    
    # Discard the image
    success = dataset_manager.discard_image("test_image.jpg")
    assert success is True
    
    # Verify image is removed from labels
    assert dataset_manager.get_labels("test_image.jpg") is None
    
    # Verify image is in discarded list
    assert dataset_manager.is_image_discarded("test_image.jpg") is True

def test_discard_nonexistent_image(dataset_manager):
    """Test attempting to discard a nonexistent image."""
    success = dataset_manager.discard_image("nonexistent.jpg")
    assert success is False

def test_discard_image_with_custom_reason(dataset_manager, test_image):
    """Test discarding an image with a custom reason."""
    # Save a test image
    image_path = os.path.join(dataset_manager.raw_dir, "test_image.jpg")
    test_image.save(image_path)
    
    # Discard with custom reason
    custom_reason = "blurry_image"
    success = dataset_manager.discard_image("test_image.jpg", reason=custom_reason)
    assert success is True
    
    # Verify reason was saved
    with open(dataset_manager.discarded_file, 'r') as f:
        discarded_data = json.load(f)
    
    assert discarded_data["discarded_images"]["test_image.jpg"]["reason"] == custom_reason 