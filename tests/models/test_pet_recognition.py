import pytest
import torch
from PIL import Image
import numpy as np

from app.models.pet_recognition.model import PetRecognitionModel

@pytest.fixture
def test_image():
    """Create a test image with specific characteristics."""
    # Create a simple test image
    width, height = 224, 224  # Match model's expected input size
    pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)

def test_model_initialization(test_pet_recognition_model):
    """Test that the model initializes correctly."""
    assert test_pet_recognition_model.model is not None
    assert test_pet_recognition_model.processor is not None
    assert test_pet_recognition_model.config is not None
    assert hasattr(test_pet_recognition_model.config, 'class_names')
    assert len(test_pet_recognition_model.config.class_names) == test_pet_recognition_model.config.num_classes

def test_model_preprocessing(test_pet_recognition_model, test_image):
    """Test image preprocessing."""
    processed = test_pet_recognition_model.preprocess(test_image)
    assert isinstance(processed, torch.Tensor)
    assert len(processed.shape) == 4  # batch, channels, height, width

def test_model_prediction(test_pet_recognition_model, test_image):
    """Test model prediction."""
    processed = test_pet_recognition_model.preprocess(test_image)
    predictions = test_pet_recognition_model.predict(processed)
    assert isinstance(predictions, dict)
    assert "predicted_class" in predictions
    assert "confidence" in predictions
    assert "probabilities" in predictions
    assert len(predictions["probabilities"]) == test_pet_recognition_model.config.num_classes

def test_confidence_levels(test_pet_recognition_model):
    """Test confidence level classification."""
    test_confidences = [0.9, 0.7, 0.4, 0.2]
    expected_levels = ["high", "medium", "low", "very low"]
    
    for conf, expected in zip(test_confidences, expected_levels):
        level = test_pet_recognition_model.get_confidence_level(conf)
        assert level == expected

def test_breed_details(test_pet_recognition_model):
    """Test breed details retrieval."""
    # Test valid breed
    valid_breed = test_pet_recognition_model.config.class_names[0]  # "labrador"
    details = test_pet_recognition_model._get_breed_details(valid_breed)
    assert isinstance(details, str)
    assert details != "No detailed information available"
    
    # Test invalid breed
    invalid_breed = "invalid_breed"
    details = test_pet_recognition_model._get_breed_details(invalid_breed)
    assert details == "No detailed information available"
    
    # Test "other" breed
    other_details = test_pet_recognition_model._get_breed_details("other")
    assert other_details == "No detailed information available"

def test_similar_breeds(test_pet_recognition_model):
    """Test similar breeds retrieval."""
    # Test valid breed
    valid_breed = test_pet_recognition_model.config.class_names[0]  # "labrador"
    similar = test_pet_recognition_model._get_similar_breeds(valid_breed)
    assert isinstance(similar, list)
    assert len(similar) == 3
    assert valid_breed not in similar
    assert all(breed in test_pet_recognition_model.config.class_names for breed in similar)
    
    # Test invalid breed
    invalid_breed = "invalid_breed"
    similar = test_pet_recognition_model._get_similar_breeds(invalid_breed)
    assert similar == []
    
    # Test "other" breed
    other_similar = test_pet_recognition_model._get_similar_breeds("other")
    assert other_similar == []

def test_end_to_end_prediction(test_pet_recognition_model, test_image):
    """Test complete prediction pipeline."""
    result = test_pet_recognition_model(test_image)
    assert isinstance(result, dict)
    assert all(key in result for key in [
        "predicted_class",
        "confidence",
        "confidence_level",
        "top_predictions",
        "breed_details",
        "similar_breeds"
    ])
    
    # Check types and values
    assert isinstance(result["predicted_class"], str)
    assert isinstance(result["confidence"], float)
    assert result["confidence_level"] in ["high", "medium", "low", "very low"]
    assert isinstance(result["top_predictions"], list)
    assert isinstance(result["breed_details"], str)
    assert isinstance(result["similar_breeds"], list)
    
    # Check top predictions structure
    for pred in result["top_predictions"]:
        assert isinstance(pred, dict)
        assert "breed" in pred
        assert "confidence" in pred
        assert isinstance(pred["breed"], str)
        assert isinstance(pred["confidence"], float)
        assert pred["breed"] in test_pet_recognition_model.config.class_names

def test_postprocessing(test_pet_recognition_model, test_image):
    """Test prediction postprocessing."""
    processed = test_pet_recognition_model.preprocess(test_image)
    predictions = test_pet_recognition_model.predict(processed)
    results = test_pet_recognition_model.postprocess(predictions)
    
    # Verificar estructura básica
    assert isinstance(results, dict)
    assert "predicted_class" in results
    assert "confidence" in results
    assert "top_predictions" in results
    assert "breed_details" in results
    assert "similar_breeds" in results
    
    # Verificar valores
    assert results["predicted_class"] in test_pet_recognition_model.config.class_names
    assert 0 <= results["confidence"] <= 1
    assert isinstance(results["breed_details"], str)
    assert isinstance(results["similar_breeds"], list)

def test_full_pipeline(test_pet_recognition_model, test_image):
    """Test the complete prediction pipeline."""
    results = test_pet_recognition_model(test_image)
    
    # Verificar estructura completa de resultados
    assert isinstance(results, dict)
    assert all(key in results for key in [
        "predicted_class",
        "confidence",
        "top_predictions",
        "breed_details",
        "similar_breeds"
    ])
    
    # Verificar tipos de datos
    assert isinstance(results["predicted_class"], str)
    assert isinstance(results["confidence"], float)
    assert isinstance(results["top_predictions"], list)
    assert isinstance(results["breed_details"], str)
    assert isinstance(results["similar_breeds"], list)
    
    # Verificar valores
    assert results["predicted_class"] in test_pet_recognition_model.config.class_names
    assert 0 <= results["confidence"] <= 1
    assert len(results["top_predictions"]) <= 5  # Ajustado a 5 predicciones principales
    assert len(results["breed_details"]) > 0
    
    # Si no es "other", debería tener razas similares
    if results["predicted_class"] != "other":
        assert len(results["similar_breeds"]) > 0 