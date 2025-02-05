import pytest
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
import io
import torch

from app.main import app
from app.core.config import settings
from app.tools.dataset_manager import DatasetManager
from app.models.pet_recognition.model import PetRecognitionModel
from app.models.disease_detection.model import DiseaseDetectionModel
from app.models.nutrition_analysis.model import NutritionAnalysisModel
from app.models.pregnancy_detection.model import PregnancyDetectionModel

MODEL_DIRS = [
    "pet_recognition",
    "disease_detection",
    "nutrition_analysis",
    "pregnancy_detection"
]

@pytest.fixture(scope="function")
def test_env():
    """Create a temporary environment for tests."""
    original_data_dir = os.getenv("DATA_DIR", "data")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set up the environment
        os.environ["DATA_DIR"] = tmp_dir
        
        # Create required directories
        datasets_dir = os.path.join(tmp_dir, "datasets")
        raw_images_dir = os.path.join(tmp_dir, "raw_images")
        processed_images_dir = os.path.join(tmp_dir, "processed_images")
        models_dir = os.path.join(tmp_dir, "models")
        
        for dir_path in [datasets_dir, raw_images_dir, processed_images_dir, models_dir]:
            os.makedirs(dir_path, exist_ok=True)
            os.chmod(dir_path, 0o777)
        
        # Create model directories and dummy weights for each model
        for model_name in MODEL_DIRS:
            weights_dir = os.path.join(models_dir, model_name, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            os.chmod(weights_dir, 0o777)
            
            # Create dummy weights file
            dummy_weights_path = os.path.join(weights_dir, "model.pt")
            
            # Create appropriate sized dummy model based on model type
            if model_name == "pregnancy_detection":
                dummy_model = torch.nn.Linear(1280, 2)  # EfficientNet output size to 2 classes
            else:
                num_classes = {
                    "pet_recognition": 10,
                    "disease_detection": 5,
                    "nutrition_analysis": 4
                }.get(model_name, 10)
                dummy_model = torch.nn.Linear(768, num_classes)  # ViT output size
            
            torch.save(dummy_model.state_dict(), dummy_weights_path)
        
        yield tmp_dir
        
        # Restore original environment
        os.environ["DATA_DIR"] = original_data_dir

@pytest.fixture
def client(test_env):
    """Create a test client with a temporary environment."""
    return TestClient(app)

@pytest.fixture
def test_image():
    """Create a test image."""
    width, height = 224, 224  # Standard size for all models
    pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    return img

@pytest.fixture
def test_db():
    """Fixture to provide a test database session."""
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def dataset_manager(test_env):
    """Create a dataset manager instance with temporary directory."""
    return DatasetManager()

@pytest.fixture
def test_pet_recognition_model(test_env):
    """Create a test instance of PetRecognitionModel with dummy weights."""
    weights_path = os.path.join(test_env, "models", "pet_recognition", "weights", "model.pt")
    return PetRecognitionModel(weights_path=weights_path)

@pytest.fixture
def test_disease_detection_model(test_env):
    """Create a test instance of DiseaseDetectionModel with dummy weights."""
    weights_path = os.path.join(test_env, "models", "disease_detection", "weights", "model.pt")
    return DiseaseDetectionModel(weights_path=weights_path)

@pytest.fixture
def test_nutrition_analysis_model(test_env):
    """Create a test instance of NutritionAnalysisModel with dummy weights."""
    weights_path = os.path.join(test_env, "models", "nutrition_analysis", "weights", "model.pt")
    return NutritionAnalysisModel(weights_path=weights_path)

@pytest.fixture
def test_pregnancy_detection_model(test_env):
    """Create a test instance of PregnancyDetectionModel with dummy weights."""
    weights_path = os.path.join(test_env, "models", "pregnancy_detection", "weights", "model.pt")
    return PregnancyDetectionModel(weights_path=weights_path) 