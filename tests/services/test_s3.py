import pytest
from PIL import Image
import io

from app.services.s3 import S3Service
from app.schemas.image import ImageType

@pytest.fixture
def s3_service():
    return S3Service()

@pytest.fixture
def test_image():
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img.format = 'JPEG'
    return img

def test_generate_file_path(s3_service):
    path = s3_service._generate_file_path(
        ImageType.DISEASE_DETECTION,
        "test.jpg"
    )
    assert path.startswith("disease_detection/")
    assert path.endswith("test.jpg")
    assert len(path.split("/")) == 2

def test_upload_and_get_image(s3_service, test_image):
    # Upload image
    success, s3_path = s3_service.upload_image(
        test_image,
        "test.jpg",
        ImageType.DISEASE_DETECTION
    )
    assert success
    assert s3_path

    # Get image back
    retrieved_image = s3_service.get_image(s3_path)
    assert retrieved_image is not None
    assert isinstance(retrieved_image, Image.Image)
    
    # Clean up
    assert s3_service.delete_image(s3_path)

def test_get_image_url(s3_service, test_image):
    # Upload image first
    success, s3_path = s3_service.upload_image(
        test_image,
        "test.jpg",
        ImageType.DISEASE_DETECTION
    )
    assert success

    # Get URL
    url = s3_service.get_image_url(s3_path)
    assert url is not None
    assert url.startswith("https://")
    
    # Clean up
    assert s3_service.delete_image(s3_path) 