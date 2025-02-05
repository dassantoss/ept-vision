#!/usr/bin/env python3
"""
API endpoints for image analysis.

This module provides endpoints for analyzing images using different models
for disease detection, pet recognition, nutrition analysis,
and pregnancy detection.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from PIL import Image
import io
import logging
import sys

from app.core.database import get_db
from app.schemas.image import ImageType
from app.schemas.prediction import (
    DiseaseDetectionPrediction,
    NutritionAnalysisPrediction,
    PregnancyDetectionPrediction,
    PetRecognitionPrediction
)
from app.services.s3 import S3Service
from app.models.disease_detection import DiseaseDetectionModel
from app.models.pet_recognition import PetRecognitionModel
from app.models.nutrition_analysis import NutritionAnalysisModel
from app.models.pregnancy_detection import PregnancyDetectionModel
from app.core.logging import get_logger

logger = get_logger("ept_vision.analysis_api")
router = APIRouter()
s3_service = S3Service()

# Initialize models
models = {
    "disease_detection": DiseaseDetectionModel(),
    "pet_recognition": PetRecognitionModel(),
    "nutrition_analysis": NutritionAnalysisModel(),
    "pregnancy_detection": PregnancyDetectionModel()
}


@router.post("/analyze/{image_type}/{image_id}")
async def analyze_image(image_type: str, image_id: str):
    """
    Analyze an image using the specified model.

    Args:
        image_type: Type of analysis to perform
        image_id: Identifier of the image to analyze

    Returns:
        Dict containing analysis results

    Raises:
        HTTPException: If image type is invalid, image not found or
                      processing error occurs
    """
    try:
        if image_type not in models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image type: {image_type}"
            )

        image_bytes = await s3_service.get_image(image_id, image_type)

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {image_id}"
            )

        image = Image.open(io.BytesIO(image_bytes))
        model = models[image_type]

        try:
            result = model(image)
            return {"result": result}
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing image: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
