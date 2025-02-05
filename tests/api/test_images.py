import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app
from app.schemas.image import ImageType
from app.core.logging import get_logger

logger = get_logger("test_images")

client = TestClient(app)

@pytest.fixture
def test_image_file():
    # Create a test image with some pattern to increase sharpness
    logger.info("Creating test image...")
    img = Image.new('RGB', (300, 300), color='white')
    # Draw some patterns to increase sharpness
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw diagonal lines
    for i in range(0, 300, 10):
        draw.line([(i, 0), (0, i)], fill='black', width=2)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def uploaded_image_path(test_image_file):
    """Fixture para subir una imagen y obtener su path"""
    response = client.post(
        f"/api/v1/images/upload/{ImageType.DISEASE_DETECTION}",
        files={"file": ("test.jpg", test_image_file, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    return data["path"]

def test_upload_image(test_image_file):
    response = client.post(
        f"/api/v1/images/upload/{ImageType.DISEASE_DETECTION}",
        files={"file": ("test.jpg", test_image_file, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    logger.info(f"Upload response: {data}")
    assert data["success"]
    assert data["path"]
    assert data["url"]

def test_get_image(uploaded_image_path):
    image_id = uploaded_image_path.split("/")[-1]
    response = client.get(
        f"/api/v1/images/image/{ImageType.DISEASE_DETECTION}/{image_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert data["url"]

def test_delete_image(uploaded_image_path):
    image_id = uploaded_image_path.split("/")[-1]
    response = client.delete(
        f"/api/v1/images/image/{ImageType.DISEASE_DETECTION}/{image_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] 