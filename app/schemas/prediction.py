#!/usr/bin/env python3
"""Prediction schemas for model outputs."""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionDetail(BaseModel):
    """Schema for individual prediction details.

    Attributes:
        class_name: Name of predicted class
        probability: Prediction probability score
    """
    class_name: str = Field(..., description="Predicted class name")
    probability: float = Field(
        ...,
        description="Prediction probability",
        ge=0,
        le=1
    )


class BasePrediction(BaseModel):
    """Base schema for model predictions.

    Attributes:
        predicted_class: Main predicted class
        confidence: Overall prediction confidence
        top_predictions: Top 3 prediction details
    """
    predicted_class: str = Field(..., description="Main predicted class")
    confidence: float = Field(
        ...,
        description="Prediction confidence",
        ge=0,
        le=1
    )
    top_predictions: List[PredictionDetail] = Field(
        ...,
        description="Top 3 predictions"
    )


class DiseaseDetectionPrediction(BasePrediction):
    """Schema for disease detection results.

    Attributes:
        disease_severity: Detected disease severity level
        recommended_action: Recommended action based on detection
    """
    disease_severity: Optional[str] = Field(
        None,
        description="Disease severity if detected"
    )
    recommended_action: Optional[str] = Field(
        None,
        description="Recommended action based on prediction"
    )


class NutritionAnalysisPrediction(BasePrediction):
    """Schema for nutrition analysis results.

    Attributes:
        bmi_estimate: Estimated body mass index
        nutrition_recommendations: List of nutritional recommendations
    """
    bmi_estimate: Optional[float] = Field(
        None,
        description="Estimated body mass index"
    )
    nutrition_recommendations: Optional[List[str]] = Field(
        None,
        description="Nutritional recommendations"
    )


class PregnancyDetectionPrediction(BasePrediction):
    """Schema for pregnancy detection results.

    Attributes:
        estimated_stage: Estimated pregnancy stage
        confidence_level: Detection confidence level
    """
    estimated_stage: Optional[str] = Field(
        None,
        description="Estimated pregnancy stage if detected"
    )
    confidence_level: Optional[str] = Field(
        None,
        description="Detection confidence level"
    )


class PetRecognitionPrediction(BasePrediction):
    """Schema for pet recognition results.

    Attributes:
        breed_details: Additional breed information
        similar_breeds: List of similar detected breeds
    """
    breed_details: Optional[str] = Field(
        None,
        description="Additional breed details"
    )
    similar_breeds: Optional[List[str]] = Field(
        None,
        description="Similar detected breeds"
    )
