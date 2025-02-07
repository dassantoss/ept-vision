#!/usr/bin/env python3
"""
API endpoints for image analysis.

This module provides endpoints for analyzing images using different models
for disease detection, pet recognition, nutrition analysis,
and pregnancy detection.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from PIL import Image
import io
import logging
import sys
import os

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
from app.models.pet_analysis import PetAnalysisModel
from app.tools.dataset_manager import DatasetManager

logger = get_logger("ept_vision.analysis_api")
router = APIRouter()
s3_service = S3Service()

# Initialize models and dataset manager
models = {
    "disease_detection": DiseaseDetectionModel(),
    "pet_recognition": PetRecognitionModel(),
    "nutrition_analysis": NutritionAnalysisModel(),
    "pregnancy_detection": PregnancyDetectionModel(),
    "pet_analysis": PetAnalysisModel()
}
dataset_manager = DatasetManager()

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
        # Normalize image type
        image_type = image_type.lower()
        
        # Map 'pet' to 'pet_analysis' for the model lookup
        model_type = 'pet_analysis' if image_type == 'pet' else image_type
        
        if model_type not in models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image type: {image_type}"
            )

        # Get image from local filesystem using dataset manager
        image = dataset_manager.get_image(image_id)
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {image_id}"
            )

        model = models[model_type]

        try:
            result = model.analyze_image(image)
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

@router.post("/analyze/pet/{image_id}", response_model=Dict[str, Any])
async def analyze_pet(
    image_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Analyze a pet image and return detailed analysis.
    
    Performs comprehensive analysis including:
    - Animal type identification (dog/cat)
    - Size estimation
    - Body condition assessment
    - Health issues detection
    - Pregnancy indicators
    - Environment/context evaluation
    
    Args:
        image_id: ID of the image to analyze
        db: Database session
        
    Returns:
        Dict containing analysis results:
        - animal_info: Basic animal information (type, size)
        - health_assessment: Health and condition evaluation
        - environment: Context/environment analysis
        - analysis_quality: Analysis quality and confidence
        
    Raises:
        HTTPException: If image is not found or analysis fails
    """
    try:
        # Get image from local filesystem using dataset manager
        image = dataset_manager.get_image(image_id)
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {image_id}"
            )
        
        # Get model instance
        model = models["pet_analysis"]
        
        try:
            # Analyze image
            raw_result = model.analyze_image(image)
            
            # Format results for end user
            return {
                "result": {
                    "animal_info": {
                        "type": raw_result["animal_type"],
                        "size": raw_result["size"],
                        "confidence": raw_result["animal_type_confidence"]
                    },
                    "health_assessment": {
                        "body_condition": raw_result["body_condition"],
                        "health_issues": raw_result["health_issues"],
                        "pregnancy_indicators": raw_result["pregnancy_indicators"],
                        "health_confidence": raw_result["health_issues_confidence"],
                        "recommendations": get_health_recommendations(raw_result)
                    },
                    "environment": {
                        "context": raw_result["context"],
                        "context_confidence": raw_result["context_confidence"],
                        "image_quality": raw_result["image_quality"]
                    },
                    "analysis_quality": {
                        "overall_confidence": raw_result["confidence"],
                        "image_quality_score": raw_result["image_quality_confidence"],
                        "analysis_limitations": get_analysis_limitations(raw_result)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pet image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error analyzing image: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_pet: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def get_health_recommendations(result: Dict[str, Any]) -> List[str]:
    """
    Generate health recommendations based on analysis results.
    
    Args:
        result: Raw analysis results from the model
        
    Returns:
        List of health recommendations based on detected conditions
    """
    recommendations = []
    
    if result["body_condition"] == "underweight":
        recommendations.append(
            "Veterinary consultation recommended for nutrition assessment"
        )
    elif result["body_condition"] == "overweight":
        recommendations.append(
            "Exercise plan and controlled diet recommended"
        )
        
    if result["health_issues"] != "none":
        recommendations.append(
            "Veterinary check-up suggested to evaluate detected health issues"
        )
        
    if result["pregnancy_indicators"] in ["possible", "visible"]:
        recommendations.append(
            "Veterinary care recommended to confirm and monitor possible pregnancy"
        )
        
    return recommendations or ["No specific recommendations required at this time"]

def calculate_overall_confidence(result: Dict[str, Any]) -> float:
    """
    Calculate overall confidence of the analysis.
    
    Args:
        result: Raw analysis results from the model
        
    Returns:
        Float representing the average confidence across all metrics
    """
    confidence_scores = [
        result["confidence"],
        result["health_confidence"],
        result["context_confidence"],
        result["quality_confidence"]
    ]
    return sum(confidence_scores) / len(confidence_scores)

def get_analysis_limitations(result: Dict[str, Any]) -> List[str]:
    """
    Identify limitations in the analysis.
    
    Args:
        result: Raw analysis results from the model
        
    Returns:
        List of identified limitations in the analysis
    """
    limitations = []
    
    if result["quality_confidence"] < 0.6:
        limitations.append("Image quality might affect analysis accuracy")
        
    if result["confidence"] < 0.7:
        limitations.append("Low confidence in animal identification")
        
    if result["health_confidence"] < 0.6:
        limitations.append("Health assessment may require veterinary confirmation")
        
    return limitations or ["No significant limitations detected in the analysis"]

@router.get("/debug/directory")
async def debug_directory():
    """
    Debug endpoint to check directory contents and permissions.
    """
    try:
        raw_dir = dataset_manager.raw_dir
        image_path = os.path.join(raw_dir, "20250206_051548_001250.jpeg")
        
        return {
            'raw_dir': raw_dir,
            'raw_dir_exists': os.path.exists(raw_dir),
            'raw_dir_is_dir': os.path.isdir(raw_dir) if os.path.exists(raw_dir) else False,
            'raw_dir_permissions': oct(os.stat(raw_dir).st_mode) if os.path.exists(raw_dir) else None,
            'image_path': image_path,
            'image_exists': os.path.exists(image_path),
            'image_permissions': oct(os.stat(image_path).st_mode) if os.path.exists(image_path) else None,
            'raw_dir_contents': os.listdir(raw_dir) if os.path.exists(raw_dir) and os.path.isdir(raw_dir) else []
        }
    except Exception as e:
        logger.error(f"Error debugging directory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error debugging directory: {str(e)}"
        )
