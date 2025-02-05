import pytest
import io
import os
import json
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.tools.dataset_manager import DatasetManager

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def test_image():
    """Create a test image."""
    # Crear una imagen de prueba
    img = Image.new('RGB', (100, 100), color='red')
    # Convertir a bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

def test_upload_image(client, test_image):
    """Test image upload endpoint."""
    response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert response.status_code == 200
    assert "filename" in response.json()

def test_upload_invalid_file(client):
    """Test uploading an invalid file."""
    response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.txt", b"invalid data", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_get_unlabeled_images(client, test_image):
    """Test getting unlabeled images."""
    # First upload an image
    upload_response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert upload_response.status_code == 200
    
    # Get unlabeled images
    response = client.get("/api/v1/dataset/unlabeled")
    assert response.status_code == 200
    assert "images" in response.json()
    assert len(response.json()["images"]) > 0

def test_update_labels(client, test_image):
    """Test updating image labels."""
    # First upload an image
    upload_response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert upload_response.status_code == 200
    image_id = upload_response.json()["filename"]
    
    # Update labels
    labels = {
        "disease": "healthy",
        "nutrition": "good",
        "pregnancy": False,
        "breed": "unknown"
    }
    response = client.post(
        f"/api/v1/dataset/labels/{image_id}",
        json=labels
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Labels updated successfully"

def test_get_labels(client, test_image):
    """Test getting image labels."""
    # First upload an image and add labels
    upload_response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert upload_response.status_code == 200
    image_id = upload_response.json()["filename"]
    
    # Add labels
    labels = {
        "disease": "healthy",
        "nutrition": "good",
        "pregnancy": False,
        "breed": "unknown"
    }
    client.post(f"/api/v1/dataset/labels/{image_id}", json=labels)
    
    # Get labels
    response = client.get(f"/api/v1/dataset/labels/{image_id}")
    assert response.status_code == 200
    assert response.json() == labels

def test_get_labels_nonexistent_image(client):
    """Test getting labels for a nonexistent image."""
    response = client.get("/api/v1/dataset/labels/nonexistent.jpg")
    assert response.status_code == 404
    assert "Image not found" in response.json()["detail"]

def test_list_images(client, test_image):
    """Test listing images endpoint."""
    # Upload a test image first
    response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert response.status_code == 200
    
    # List images
    response = client.get("/api/v1/dataset/images")
    assert response.status_code == 200
    assert isinstance(response.json()["images"], list)
    assert len(response.json()["images"]) > 0

def test_get_image_labels(client, test_image):
    """Test getting all image labels."""
    # Upload and label test images
    upload_response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert upload_response.status_code == 200
    image_id = upload_response.json()["filename"]
    
    # Add labels
    labels = {
        "disease": "healthy",
        "nutrition": "good",
        "pregnancy": False,
        "breed": "unknown"
    }
    client.post(f"/api/v1/dataset/labels/{image_id}", json=labels)
    
    # Get all labels
    response = client.get("/api/v1/dataset/labels")
    assert response.status_code == 200
    assert isinstance(response.json()["labels"], dict)
    assert image_id in response.json()["labels"]

def test_save_image_labels(client, test_image):
    """Test saving multiple image labels."""
    # Upload test images
    response1 = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test1.png", test_image, "image/png")}
    )
    response2 = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test2.png", test_image, "image/png")}
    )
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    image_id1 = response1.json()["filename"]
    image_id2 = response2.json()["filename"]
    
    # Save labels for both images
    labels = {
        image_id1: {
            "disease": "healthy",
            "nutrition": "good"
        },
        image_id2: {
            "disease": "sick",
            "nutrition": "poor"
        }
    }
    response = client.post("/api/v1/dataset/labels", json=labels)
    assert response.status_code == 200
    assert response.json()["message"] == "Labels saved successfully"

def test_get_labeling_progress(client, test_image):
    """Test getting labeling progress."""
    # Upload and label test images
    upload_response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert upload_response.status_code == 200
    image_id = upload_response.json()["filename"]
    
    # Add labels
    labels = {
        "disease": "healthy",
        "nutrition": "good"
    }
    client.post(f"/api/v1/dataset/labels/{image_id}", json=labels)
    
    # Get progress
    response = client.get("/api/v1/dataset/progress")
    assert response.status_code == 200
    assert "total_images" in response.json()
    assert "labeled_images" in response.json()
    assert response.json()["labeled_images"] > 0

@pytest.mark.integration
def test_full_labeling_flow(client, test_image):
    """Test complete labeling workflow."""
    # 1. Upload image
    upload_response = client.post(
        "/api/v1/dataset/upload",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert upload_response.status_code == 200
    image_id = upload_response.json()["filename"]
    
    # 2. Get unlabeled images
    unlabeled_response = client.get("/api/v1/dataset/unlabeled")
    assert unlabeled_response.status_code == 200
    assert image_id in [img["id"] for img in unlabeled_response.json()["images"]]
    
    # 3. Add labels
    labels = {
        "disease": "healthy",
        "nutrition": "good",
        "pregnancy": False,
        "breed": "unknown"
    }
    label_response = client.post(f"/api/v1/dataset/labels/{image_id}", json=labels)
    assert label_response.status_code == 200
    
    # 4. Verify labels
    verify_response = client.get(f"/api/v1/dataset/labels/{image_id}")
    assert verify_response.status_code == 200
    assert verify_response.json() == labels
    
    # 5. Check progress
    progress_response = client.get("/api/v1/dataset/progress")
    assert progress_response.status_code == 200
    assert progress_response.json()["labeled_images"] > 0 