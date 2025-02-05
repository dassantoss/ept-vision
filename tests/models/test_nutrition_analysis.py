import pytest
import torch
from PIL import Image
import numpy as np
import os

from app.models.nutrition_analysis.model import NutritionAnalysisModel
from app.core.model_config import NUTRITION_ANALYSIS_CONFIG

@pytest.fixture
def nutrition_model():
    return NutritionAnalysisModel()

@pytest.fixture
def test_image():
    """Create a test image."""
    width, height = 224, 224  # Match model's expected input size
    pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)

@pytest.fixture
def test_nutrition_analysis_model(test_env):
    """Create a test instance of NutritionAnalysisModel with dummy weights."""
    weights_path = os.path.join(test_env, "models", "nutrition_analysis", "weights", "model.pt")
    model = NutritionAnalysisModel()
    model.config.weights_path = weights_path
    return model

def test_model_initialization(test_nutrition_analysis_model):
    """Test that the model initializes correctly."""
    assert test_nutrition_analysis_model.model is not None
    assert test_nutrition_analysis_model.processor is not None
    assert test_nutrition_analysis_model.config is not None
    assert hasattr(test_nutrition_analysis_model.config, 'nutrition_states')
    assert len(test_nutrition_analysis_model.config.nutrition_states) == test_nutrition_analysis_model.config.num_classes

def test_model_preprocessing(test_nutrition_analysis_model, test_image):
    """Test image preprocessing."""
    processed = test_nutrition_analysis_model.preprocess(test_image)
    assert isinstance(processed, torch.Tensor)
    assert len(processed.shape) == 4  # batch, channels, height, width
    assert processed.shape[1:] == (3, 224, 224)  # RGB image with expected size

def test_model_prediction(test_nutrition_analysis_model, test_image):
    """Test model prediction."""
    processed = test_nutrition_analysis_model.preprocess(test_image)
    predictions = test_nutrition_analysis_model.predict(processed)
    
    assert isinstance(predictions, dict)
    assert "predicted_class" in predictions
    assert "confidence" in predictions
    assert isinstance(predictions["predicted_class"], int)
    assert isinstance(predictions["confidence"], float)
    assert 0 <= predictions["confidence"] <= 1
    assert predictions["predicted_class"] in test_nutrition_analysis_model.config.nutrition_states

def test_confidence_levels(test_nutrition_analysis_model):
    """Test confidence level classification."""
    test_confidences = [0.9, 0.7, 0.55, 0.4]
    expected_levels = ["high", "medium", "low", "very low"]
    
    for conf, expected in zip(test_confidences, expected_levels):
        level = test_nutrition_analysis_model.get_confidence_level(conf)
        assert level == expected

def test_bmi_estimation(test_nutrition_analysis_model):
    """Test BMI range estimation."""
    test_cases = [
        ("underweight", (0, 18.5)),
        ("normal", (18.5, 24.9)),
        ("overweight", (25, 29.9)),
        ("obese", (30, float('inf')))
    ]
    
    for state, expected_range in test_cases:
        bmi_range = test_nutrition_analysis_model.estimate_bmi(state)
        assert bmi_range == expected_range

def test_nutrition_recommendations(test_nutrition_analysis_model):
    """Test nutrition recommendations."""
    for state in ["underweight", "normal", "overweight", "obese"]:
        recommendations = test_nutrition_analysis_model.get_nutrition_recommendations(state)
        assert isinstance(recommendations, str)
        assert len(recommendations) > 0
        
    # Test invalid state
    invalid_recommendations = test_nutrition_analysis_model.get_nutrition_recommendations("invalid_state")
    assert "consult" in invalid_recommendations.lower()

def test_postprocessing(test_nutrition_analysis_model):
    """Test prediction postprocessing."""
    mock_predictions = {
        "predicted_class": 1,  # normal
        "confidence": 0.85
    }
    
    results = test_nutrition_analysis_model.postprocess(mock_predictions)
    
    assert isinstance(results, dict)
    assert all(key in results for key in [
        "nutrition_state",
        "confidence",
        "confidence_level",
        "estimated_bmi_range",
        "recommendations"
    ])
    
    assert results["nutrition_state"] in test_nutrition_analysis_model.config.nutrition_states.values()
    assert isinstance(results["confidence"], float)
    assert results["confidence_level"] in ["high", "medium", "low", "very low"]
    assert isinstance(results["estimated_bmi_range"], tuple)
    assert isinstance(results["recommendations"], str)

def test_end_to_end_prediction(test_nutrition_analysis_model, test_image):
    """Test complete prediction pipeline."""
    result = test_nutrition_analysis_model(test_image)
    
    assert isinstance(result, dict)
    assert all(key in result for key in [
        "nutrition_state",
        "confidence",
        "confidence_level",
        "estimated_bmi_range",
        "recommendations"
    ])
    
    assert result["nutrition_state"] in test_nutrition_analysis_model.config.nutrition_states.values()
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1
    assert result["confidence_level"] in ["high", "medium", "low", "very low"]
    assert isinstance(result["estimated_bmi_range"], tuple)
    assert isinstance(result["recommendations"], str)

def test_invalid_inputs(test_nutrition_analysis_model):
    """Test handling of invalid inputs."""
    # Test with invalid image type
    with pytest.raises(ValueError, match="Input must be a PIL Image"):
        test_nutrition_analysis_model.preprocess(np.zeros((224, 224, 3)))
    
    # Test with empty image
    empty_image = Image.new('RGB', (224, 224), color='black')
    result = test_nutrition_analysis_model(empty_image)
    assert isinstance(result, dict)
    assert "nutrition_state" in result
    assert "confidence" in result

def test_model_consistency(test_nutrition_analysis_model, test_image):
    """Test model consistency with multiple predictions."""
    predictions = [test_nutrition_analysis_model(test_image) for _ in range(3)]
    
    first_prediction = predictions[0]["nutrition_state"]
    first_confidence = predictions[0]["confidence"]
    
    for pred in predictions[1:]:
        assert pred["nutrition_state"] == first_prediction
        assert abs(pred["confidence"] - first_confidence) < 1e-6 