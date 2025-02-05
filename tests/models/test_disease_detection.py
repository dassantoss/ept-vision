import pytest
import torch
from PIL import Image
import numpy as np
import os

from app.models.disease_detection.model import DiseaseDetectionModel
from app.core.model_config import DISEASE_DETECTION_CONFIG

@pytest.fixture
def disease_model():
    return DiseaseDetectionModel()

@pytest.fixture
def test_images():
    """Create test images for different conditions."""
    width, height = DISEASE_DETECTION_CONFIG.input_size
    images = {}
    
    # Healthy image (normal skin color)
    healthy_img = np.ones((height, width, 3), dtype=np.uint8) * 220  # Light skin tone
    images["healthy"] = Image.fromarray(healthy_img)
    
    # Skin infection image (reddish patches)
    skin_infection_img = np.ones((height, width, 3), dtype=np.uint8) * 220
    # Add reddish patches
    skin_infection_img[50:150, 50:150] = [255, 150, 150]
    images["skin_infection"] = Image.fromarray(skin_infection_img)
    
    # Eye infection image (redness around eye area)
    eye_infection_img = np.ones((height, width, 3), dtype=np.uint8) * 220
    # Add eye area with infection
    eye_infection_img[80:140, 90:170] = [255, 160, 160]
    images["eye_infection"] = Image.fromarray(eye_infection_img)
    
    # External wounds image (dark patches with redness)
    wounds_img = np.ones((height, width, 3), dtype=np.uint8) * 220
    # Add wound-like features
    wounds_img[100:150, 100:150] = [180, 100, 100]
    wounds_img[120:140, 120:140] = [100, 50, 50]
    images["external_wounds"] = Image.fromarray(wounds_img)
    
    return images

def test_model_initialization(disease_model):
    """Test that the model initializes correctly."""
    assert disease_model.config == DISEASE_DETECTION_CONFIG
    assert disease_model.model is not None
    assert str(disease_model.device) in ['cpu', 'cuda']

def test_preprocessing(disease_model, test_images):
    """Test image preprocessing."""
    for condition, image in test_images.items():
        processed = disease_model.preprocess(image)
        assert isinstance(processed, torch.Tensor)
        assert len(processed.shape) == 4  # batch, channels, height, width
        assert processed.shape[1:] == (3, *DISEASE_DETECTION_CONFIG.input_size)

def test_prediction(disease_model, test_images):
    """Test model prediction."""
    for condition, image in test_images.items():
        processed = disease_model.preprocess(image)
        predictions = disease_model.predict(processed)
        assert isinstance(predictions, dict)
        assert "predicted_class" in predictions
        assert "confidence" in predictions
        assert "probabilities" in predictions
        assert len(predictions["probabilities"]) == disease_model.config.num_classes

def test_postprocessing(disease_model, test_images):
    """Test prediction postprocessing."""
    for condition, image in test_images.items():
        processed = disease_model.preprocess(image)
        predictions = disease_model.predict(processed)
        results = disease_model.postprocess(predictions)
        
        # Verificar estructura básica
        assert isinstance(results, dict)
        assert "predicted_class" in results
        assert "confidence" in results
        assert "confidence_level" in results
        assert "severity" in results
        assert "recommendations" in results
        assert "top_predictions" in results
        
        # Verificar valores
        assert results["predicted_class"] in disease_model.config.class_names
        assert 0 <= results["confidence"] <= 1
        assert results["confidence_level"] in ["high", "medium", "low", "very low"]
        assert isinstance(results["severity"], str)
        assert isinstance(results["recommendations"], str)
        
        # Verificar predicciones principales
        assert isinstance(results["top_predictions"], list)
        for pred in results["top_predictions"]:
            assert isinstance(pred, dict)
            assert "condition" in pred
            assert "confidence" in pred
            assert pred["condition"] in disease_model.config.class_names

def test_full_pipeline(disease_model, test_images):
    """Test the complete prediction pipeline."""
    for condition, image in test_images.items():
        results = disease_model(image)
        
        # Verificar estructura completa de resultados
        assert isinstance(results, dict)
        assert all(key in results for key in [
            "predicted_class",
            "confidence",
            "confidence_level",
            "severity",
            "recommendations",
            "top_predictions"
        ])
        
        # Verificar tipos de datos
        assert isinstance(results["predicted_class"], str)
        assert isinstance(results["confidence"], float)
        assert results["confidence_level"] in ["high", "medium", "low", "very low"]
        assert isinstance(results["severity"], str)
        assert isinstance(results["recommendations"], str)
        assert isinstance(results["top_predictions"], list)
        
        # Verificar valores
        assert results["predicted_class"] in disease_model.config.class_names
        assert 0 <= results["confidence"] <= 1
        
        # Verificar predicciones principales
        for pred in results["top_predictions"]:
            assert isinstance(pred, dict)
            assert "condition" in pred
            assert "confidence" in pred
            assert pred["condition"] in disease_model.config.class_names

def test_model_consistency(disease_model, test_images):
    """Test model consistency with multiple predictions."""
    # Realizar múltiples predicciones para cada imagen
    for condition, image in test_images.items():
        predictions = [disease_model(image) for _ in range(3)]
        
        # Verificar que las predicciones son consistentes
        first_prediction = predictions[0]["predicted_class"]
        first_confidence = predictions[0]["confidence"]
        
        for pred in predictions[1:]:
            assert pred["predicted_class"] == first_prediction
            assert abs(pred["confidence"] - first_confidence) < 1e-6

def test_error_handling(disease_model):
    """Test error handling with invalid inputs."""
    # Test con una imagen de tamaño incorrecto
    invalid_size_image = Image.new('RGB', (100, 100), color='white')
    processed = disease_model.preprocess(invalid_size_image)
    assert processed.shape[2:] == DISEASE_DETECTION_CONFIG.input_size
    
    # Test con una imagen en escala de grises
    grayscale_image = Image.new('L', DISEASE_DETECTION_CONFIG.input_size, color=128)
    processed = disease_model.preprocess(grayscale_image)
    assert processed.shape[1] == 3  # Debería convertirse a RGB

@pytest.fixture
def test_image():
    """Create a test image with specific characteristics."""
    width, height = 224, 224  # Match model's expected input size
    pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)

@pytest.fixture
def test_disease_detection_model(test_env):
    """Create a test instance of DiseaseDetectionModel with dummy weights."""
    weights_path = os.path.join(test_env, "models", "disease_detection", "weights", "model.pt")
    return DiseaseDetectionModel(weights_path=weights_path)

def test_model_initialization(test_disease_detection_model):
    """Test that the model initializes correctly."""
    assert test_disease_detection_model.model is not None
    assert test_disease_detection_model.processor is not None
    assert test_disease_detection_model.config is not None
    assert hasattr(test_disease_detection_model.config, 'class_names')
    assert len(test_disease_detection_model.config.class_names) == test_disease_detection_model.config.num_classes

def test_model_preprocessing(test_disease_detection_model, test_image):
    """Test image preprocessing."""
    processed = test_disease_detection_model.preprocess(test_image)
    assert isinstance(processed, torch.Tensor)
    assert len(processed.shape) == 4  # batch, channels, height, width

def test_model_prediction(test_disease_detection_model, test_image):
    """Test model prediction."""
    processed = test_disease_detection_model.preprocess(test_image)
    predictions = test_disease_detection_model.predict(processed)
    assert isinstance(predictions, dict)
    assert "predicted_class" in predictions
    assert "confidence" in predictions
    assert "probabilities" in predictions
    assert len(predictions["probabilities"]) == test_disease_detection_model.config.num_classes

def test_confidence_levels(test_disease_detection_model):
    """Test confidence level classification."""
    test_confidences = [0.9, 0.7, 0.4, 0.2]
    expected_levels = ["high", "medium", "low", "very low"]
    
    for conf, expected in zip(test_confidences, expected_levels):
        level = test_disease_detection_model.get_confidence_level(conf)
        assert level == expected

def test_severity_estimation(test_disease_detection_model):
    """Test severity level estimation."""
    # Test healthy case
    severity = test_disease_detection_model.estimate_severity("healthy", 0.9)
    assert severity == "none"
    
    # Test different conditions with varying confidences
    test_cases = [
        ("infection", 0.9, "severe"),
        ("infection", 0.7, "moderate"),
        ("infection", 0.4, "mild"),
        ("tumor", 0.9, "malignant"),
        ("tumor", 0.7, "suspicious"),
        ("tumor", 0.4, "benign"),
    ]
    
    for condition, confidence, expected_severity in test_cases:
        severity = test_disease_detection_model.estimate_severity(condition, confidence)
        assert severity == expected_severity

def test_treatment_recommendations(test_disease_detection_model):
    """Test treatment recommendations."""
    # Test healthy case
    recommendations = test_disease_detection_model.get_treatment_recommendations("healthy", "none")
    assert "check-ups" in recommendations.lower()
    
    # Test various conditions and severities
    test_cases = [
        ("infection", "mild"),
        ("infection", "severe"),
        ("tumor", "benign"),
        ("tumor", "malignant"),
        ("injury", "minor"),
        ("injury", "severe"),
    ]
    
    for condition, severity in test_cases:
        recommendations = test_disease_detection_model.get_treatment_recommendations(condition, severity)
        assert isinstance(recommendations, str)
        assert len(recommendations) > 0
        assert recommendations != "No recommendations available"

def test_end_to_end_prediction(test_disease_detection_model, test_image):
    """Test complete prediction pipeline."""
    result = test_disease_detection_model(test_image)
    assert isinstance(result, dict)
    assert all(key in result for key in [
        "predicted_class",
        "confidence",
        "confidence_level",
        "severity",
        "recommendations",
        "top_predictions"
    ])
    
    # Check types and values
    assert isinstance(result["predicted_class"], str)
    assert isinstance(result["confidence"], float)
    assert result["confidence_level"] in ["high", "medium", "low", "very low"]
    assert isinstance(result["severity"], str)
    assert isinstance(result["recommendations"], str)
    assert isinstance(result["top_predictions"], list)
    
    # Check top predictions structure
    for pred in result["top_predictions"]:
        assert isinstance(pred, dict)
        assert "condition" in pred
        assert "confidence" in pred
        assert isinstance(pred["condition"], str)
        assert isinstance(pred["confidence"], float)
        assert pred["condition"] in test_disease_detection_model.config.class_names

def test_invalid_condition(test_disease_detection_model):
    """Test handling of invalid conditions."""
    invalid_condition = "invalid_condition"
    
    # Test severity estimation
    severity = test_disease_detection_model.estimate_severity(invalid_condition, 0.9)
    assert severity == "unknown"
    
    # Test recommendations
    recommendations = test_disease_detection_model.get_treatment_recommendations(invalid_condition, "severe")
    assert "consult" in recommendations.lower()

def test_grayscale_handling(test_disease_detection_model):
    """Test handling of grayscale images."""
    # Create a grayscale image
    width, height = 224, 224
    pixels = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    grayscale_image = Image.fromarray(pixels, mode='L')
    
    # Process the grayscale image
    processed = test_disease_detection_model.preprocess(grayscale_image)
    assert isinstance(processed, torch.Tensor)
    assert len(processed.shape) == 4  # batch, channels, height, width
    assert processed.shape[1] == 3  # Should be converted to RGB

def test_invalid_inputs(test_disease_detection_model):
    """Test handling of invalid inputs."""
    # Test with invalid image type
    with pytest.raises(ValueError, match="Input must be a PIL Image"):
        test_disease_detection_model.preprocess(np.zeros((224, 224, 3)))
    
    # Test with empty image
    empty_image = Image.new('RGB', (224, 224), color='black')
    result = test_disease_detection_model(empty_image)
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert "confidence" in result

def test_model_consistency(test_disease_detection_model, test_image):
    """Test model consistency with multiple predictions."""
    # Make multiple predictions on the same image
    predictions = [test_disease_detection_model(test_image) for _ in range(3)]
    
    # Verify that predictions are consistent
    first_prediction = predictions[0]["predicted_class"]
    first_confidence = predictions[0]["confidence"]
    
    for pred in predictions[1:]:
        assert pred["predicted_class"] == first_prediction
        assert abs(pred["confidence"] - first_confidence) < 1e-6  # Allow small floating-point differences 