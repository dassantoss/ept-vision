import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np

from app.main import app
from app.schemas.image import ImageType
from app.services.s3 import S3Service

client = TestClient(app)

@pytest.fixture
def s3_service():
    return S3Service()

@pytest.fixture
def test_images():
    """Create test images for each type of analysis."""
    width, height = 224, 224
    images = {}
    
    # Disease detection image (simulating skin condition)
    disease_img = np.ones((height, width, 3), dtype=np.uint8) * 200
    disease_img[50:150, 50:150] = [255, 100, 100]  # Reddish patch
    images["disease_detection"] = Image.fromarray(disease_img)
    
    # Pet recognition image (simulating dog features)
    pet_img = np.ones((height, width, 3), dtype=np.uint8) * 180
    pet_img[100:120, 80:140] = [50, 50, 50]  # Dark patch for nose
    images["pet_recognition"] = Image.fromarray(pet_img)
    
    # Nutrition analysis image (simulating body profile)
    nutrition_img = np.ones((height, width, 3), dtype=np.uint8) * 190
    nutrition_img[40:200, 60:180] = [170, 170, 170]  # Body silhouette
    images["nutrition_analysis"] = Image.fromarray(nutrition_img)
    
    # Pregnancy detection image (simulating ultrasound)
    pregnancy_img = np.zeros((height, width, 3), dtype=np.uint8)
    pregnancy_img[80:160, 80:160] = np.random.randint(150, 200, (80, 80, 3))
    images["pregnancy_detection"] = Image.fromarray(pregnancy_img)
    
    return images

@pytest.fixture
def uploaded_images(s3_service, test_images):
    """Upload test images and return their IDs."""
    image_ids = {}
    
    for image_type, image in test_images.items():
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Upload to S3
        success, s3_path = s3_service.upload_image(
            image,
            f"test_{image_type}.jpg",
            ImageType(image_type),
            "image/jpeg"
        )
        
        if success:
            image_ids[image_type] = s3_path.split('/')[-1]
    
    return image_ids

def test_analyze_disease_detection(uploaded_images):
    """Test disease detection analysis."""
    image_id = uploaded_images["disease_detection"]
    response = client.post(f"/api/v1/analysis/analyze/disease_detection/{image_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "disease_severity" in data
    assert "recommended_action" in data

def test_analyze_pet_recognition(uploaded_images):
    """Test pet recognition analysis."""
    image_id = uploaded_images["pet_recognition"]
    response = client.post(f"/api/v1/analysis/analyze/pet_recognition/{image_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "breed_details" in data
    assert "similar_breeds" in data

def test_analyze_nutrition(uploaded_images):
    """Test nutrition analysis."""
    image_id = uploaded_images["nutrition_analysis"]
    response = client.post(f"/api/v1/analysis/analyze/nutrition_analysis/{image_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "bmi_estimate" in data
    assert "nutrition_recommendations" in data

def test_analyze_pregnancy(uploaded_images):
    """Test pregnancy detection analysis."""
    image_id = uploaded_images["pregnancy_detection"]
    response = client.post(f"/api/v1/analysis/analyze/pregnancy_detection/{image_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "confidence_level" in data
    assert "estimated_stage" in data

def test_analyze_invalid_image():
    """Test analysis with invalid image ID."""
    response = client.post("/api/v1/analysis/analyze/disease_detection/invalid_id")
    assert response.status_code == 404

def test_analyze_all_types(uploaded_images):
    """Test analysis with all image types."""
    for image_type, image_id in uploaded_images.items():
        response = client.post(f"/api/v1/analysis/analyze/{image_type}/{image_id}")
        assert response.status_code == 200
        data = response.json()
        
        # Common fields
        assert "predicted_class" in data
        assert "confidence" in data
        assert "top_predictions" in data
        
        # Type-specific fields
        if image_type == "disease_detection":
            assert "disease_severity" in data
            assert "recommended_action" in data
        elif image_type == "pet_recognition":
            assert "breed_details" in data
            assert "similar_breeds" in data
        elif image_type == "nutrition_analysis":
            assert "bmi_estimate" in data
            assert "nutrition_recommendations" in data
        elif image_type == "pregnancy_detection":
            assert "confidence_level" in data
            assert "estimated_stage" in data 