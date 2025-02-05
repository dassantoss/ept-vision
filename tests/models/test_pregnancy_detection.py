import pytest
import torch
from PIL import Image
import numpy as np
import os

from app.models.pregnancy_detection.model import PregnancyDetectionModel
from app.core.model_config import PREGNANCY_DETECTION_CONFIG

@pytest.fixture
def pregnancy_model():
    """Create a pregnancy detection model instance."""
    return PregnancyDetectionModel()

@pytest.fixture
def test_image():
    """Create a test image."""
    width, height = 224, 224  # Match model's expected input size
    pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)

@pytest.fixture
def test_pregnancy_detection_model(test_env):
    """Create a test instance of PregnancyDetectionModel with dummy weights."""
    weights_path = os.path.join(test_env, "models", "pregnancy_detection", "weights", "model.pt")
    model = PregnancyDetectionModel()
    model.config.weights_path = weights_path
    return model

def test_model_initialization(test_pregnancy_detection_model):
    """Test that the model initializes correctly."""
    assert test_pregnancy_detection_model.model is not None
    assert test_pregnancy_detection_model.transform is not None
    assert test_pregnancy_detection_model.config is not None
    assert hasattr(test_pregnancy_detection_model.config, 'pregnancy_stages')
    assert len(test_pregnancy_detection_model.config.pregnancy_stages) == test_pregnancy_detection_model.config.num_classes

def test_model_preprocessing(test_pregnancy_detection_model, test_image):
    """Test image preprocessing."""
    processed = test_pregnancy_detection_model.preprocess(test_image)
    assert isinstance(processed, torch.Tensor)
    assert len(processed.shape) == 4  # batch, channels, height, width
    assert processed.shape[1:] == (3, 224, 224)  # RGB image with expected size

def test_model_prediction(test_pregnancy_detection_model, test_image):
    """Test model prediction."""
    processed = test_pregnancy_detection_model.preprocess(test_image)
    predictions = test_pregnancy_detection_model.predict(processed)
    
    assert isinstance(predictions, dict)
    assert "predicted_class" in predictions
    assert "confidence" in predictions
    assert isinstance(predictions["predicted_class"], int)
    assert isinstance(predictions["confidence"], float)
    assert 0 <= predictions["confidence"] <= 1
    assert predictions["predicted_class"] in test_pregnancy_detection_model.config.pregnancy_stages

def test_confidence_levels(test_pregnancy_detection_model):
    """Test confidence level classification."""
    test_confidences = [0.9, 0.7, 0.55, 0.4]
    expected_levels = ["high", "medium", "low", "very low"]
    
    for conf, expected in zip(test_confidences, expected_levels):
        level = test_pregnancy_detection_model.get_confidence_level(conf)
        assert level == expected

def test_pregnancy_stage_estimation(test_pregnancy_detection_model):
    """Test pregnancy stage estimation."""
    test_cases = [
        (0.9, "Pregnant"),
        (0.7, "Pregnant"),
        (0.4, "Not pregnant"),
        (0.2, "Not pregnant")
    ]
    
    for confidence, expected_stage in test_cases:
        stage = test_pregnancy_detection_model.estimate_pregnancy_stage(confidence)
        assert stage == expected_stage

def test_postprocessing(test_pregnancy_detection_model):
    """Test prediction postprocessing."""
    test_cases = [
        {"predicted_class": 1, "confidence": 0.85},  # Pregnant with high confidence
        {"predicted_class": 0, "confidence": 0.75},  # Not pregnant with medium confidence
    ]
    
    for predictions in test_cases:
        results = test_pregnancy_detection_model.postprocess(predictions)
        
        assert isinstance(results, dict)
        assert all(key in results for key in [
            "is_pregnant",
            "confidence",
            "confidence_level",
            "pregnancy_stage"
        ])
        
        assert isinstance(results["is_pregnant"], bool)
        assert isinstance(results["confidence"], float)
        assert results["confidence_level"] in ["high", "medium", "low", "very low"]
        assert results["pregnancy_stage"] in test_pregnancy_detection_model.config.pregnancy_stages.values()

def test_end_to_end_prediction(test_pregnancy_detection_model, test_image):
    """Test complete prediction pipeline."""
    result = test_pregnancy_detection_model(test_image)
    
    assert isinstance(result, dict)
    assert all(key in result for key in [
        "is_pregnant",
        "confidence",
        "confidence_level",
        "pregnancy_stage"
    ])
    
    assert isinstance(result["is_pregnant"], bool)
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1
    assert result["confidence_level"] in ["high", "medium", "low", "very low"]
    assert result["pregnancy_stage"] in test_pregnancy_detection_model.config.pregnancy_stages.values()

def test_invalid_inputs(test_pregnancy_detection_model):
    """Test handling of invalid inputs."""
    # Test with invalid image type
    with pytest.raises(ValueError, match="Input must be a PIL Image"):
        test_pregnancy_detection_model.preprocess(np.zeros((224, 224, 3)))
    
    # Test with empty image
    empty_image = Image.new('RGB', (224, 224), color='black')
    result = test_pregnancy_detection_model(empty_image)
    assert isinstance(result, dict)
    assert "is_pregnant" in result
    assert "confidence" in result

def test_model_consistency(test_pregnancy_detection_model, test_image):
    """Test model consistency with multiple predictions."""
    predictions = [test_pregnancy_detection_model(test_image) for _ in range(3)]
    
    first_prediction = predictions[0]["is_pregnant"]
    first_confidence = predictions[0]["confidence"]
    
    for pred in predictions[1:]:
        assert pred["is_pregnant"] == first_prediction
        assert abs(pred["confidence"] - first_confidence) < 1e-6 