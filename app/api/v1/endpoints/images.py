#!/usr/bin/env python3
"""
Image handling endpoints for the API.

This module provides endpoints for image upload, retrieval, deletion
and analysis using various machine learning models.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from PIL import Image
import io
from io import BytesIO
import traceback

from app.core.database import get_db
from app.services.s3 import S3Service
from app.schemas.image import (
    ImageType,
    ImageUploadResponse,
    ImageQuality,
    ImageValidation
)
from app.utils.image_processing import (
    validate_image,
    assess_image_quality,
    enhance_image_quality
)
from app.models.disease_detection.model import DiseaseDetectionModel
from app.models.pet_recognition.model import PetRecognitionModel
from app.models.nutrition_analysis.model import NutritionAnalysisModel
from app.models.pregnancy_detection.model import PregnancyDetectionModel
from app.core.logging import get_logger
from app.core.auth import get_current_user
from app.models.user import User


logger = get_logger("images")
router = APIRouter()
s3_service = S3Service()


# Initialize models
disease_model = DiseaseDetectionModel()
pet_model = PetRecognitionModel()
nutrition_model = NutritionAnalysisModel()
pregnancy_model = PregnancyDetectionModel()


@router.post("/upload/{image_type}", response_model=ImageUploadResponse)
async def upload_image(
    image_type: ImageType,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> ImageUploadResponse:
    """
    Upload and process an image for a specific analysis type.

    Args:
        image_type: Type of analysis to perform
        file: Image file to upload
        db: Database session

    Returns:
        ImageUploadResponse: Upload result with path and URL if successful

    Raises:
        HTTPException: If upload or processing fails
    """
    try:
        logger.info(f"Processing upload for image type: {image_type}")
        contents = await file.read()
        logger.debug(f"File read, size: {len(contents)} bytes")

        image = Image.open(io.BytesIO(contents))
        image.format = 'JPEG'
        logger.debug(f"Image opened, size: {image.size}, mode: {image.mode}")

        validation_params = ImageValidation()
        is_valid, message = validate_image(image, validation_params)
        logger.info(f"Validation result: valid={is_valid}, message={message}")

        if not is_valid:
            return ImageUploadResponse(success=False, error=message)

        quality = assess_image_quality(image)
        logger.debug(f"Quality assessment: {quality}")

        quality_threshold = {
            'brightness': 0.2,
            'contrast': 0.3,
            'sharpness': 0.0
        }

        quality_check = (
            quality.brightness < quality_threshold['brightness'] or
            quality.contrast < quality_threshold['contrast'] or
            quality.sharpness < quality_threshold['sharpness']
        )

        if quality_check:
            return ImageUploadResponse(
                success=False,
                error=(
                    "Image quality does not meet minimum requirements: "
                    f"brightness={quality.brightness}, "
                    f"contrast={quality.contrast}, "
                    f"sharpness={quality.sharpness}"
                )
            )

        logger.info(f"Uploading to S3: filename={file.filename}")
        success, s3_path = s3_service.upload_image(
            image,
            file.filename,
            image_type,
            file.content_type or "image/jpeg"
        )

        if not success:
            logger.error(f"S3 upload failed: {s3_path}")
            return ImageUploadResponse(success=False, error=s3_path)

        url = s3_service.get_image_url(s3_path)
        logger.debug(f"Generated URL: {url}")

        return ImageUploadResponse(
            success=True,
            path=s3_path,
            url=url
        )

    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        logger.error(traceback.format_exc())
        return ImageUploadResponse(success=False, error=str(e))


@router.get("/image/{image_type}/{image_id}", response_model=ImageUploadResponse)
async def get_image(
    image_type: ImageType,
    image_id: str,
    db: Session = Depends(get_db)
) -> ImageUploadResponse:
    """
    Get image details and generate a temporary URL.

    Args:
        image_type: Type of image to retrieve
        image_id: ID of the image
        db: Database session

    Returns:
        ImageUploadResponse: Image details including temporary URL

    Raises:
        HTTPException: If image not found or retrieval fails
    """
    try:
        s3_path = f"{image_type.value}/{image_id}"
        url = s3_service.get_image_url(s3_path)

        if not url:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )

        return ImageUploadResponse(
            success=True,
            path=s3_path,
            url=url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.delete("/image/{image_type}/{image_id}")
async def delete_image(
    image_type: ImageType,
    image_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete an image from storage.

    Args:
        image_type: Type of image to delete
        image_id: ID of the image
        db: Database session

    Returns:
        Dict containing operation success status

    Raises:
        HTTPException: If deletion fails
    """
    try:
        s3_path = f"{image_type.value}/{image_id}"
        success = s3_service.delete_image(s3_path)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Image not found or could not be deleted"
            )

        return {"success": True, "message": "Image deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/analyze/{image_type}/{image_id}", response_model=Any)
async def analyze_image(
    image_type: ImageType,
    image_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analyze an image using the specified model.

    Args:
        image_type: Type of analysis to perform
        image_id: ID of the image to analyze
        current_user: Current authenticated user

    Returns:
        Dict containing analysis results

    Raises:
        HTTPException: If analysis fails or image not found
    """
    try:
        image_bytes = s3.get_object(image_id)
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Image not found")

        image = Image.open(BytesIO(image_bytes))

        if image_type == ImageType.DISEASE_DETECTION:
            result = disease_model.predict_with_cache(image, image_id)
        elif image_type == ImageType.PET_RECOGNITION:
            result = pet_model.predict_with_cache(image, image_id)
        elif image_type == ImageType.NUTRITION_ANALYSIS:
            result = nutrition_model.predict_with_cache(image, image_id)
        elif image_type == ImageType.PREGNANCY_DETECTION:
            result = pregnancy_model.predict_with_cache(image, image_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported analysis type: {image_type}"
            )

        return result

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )
