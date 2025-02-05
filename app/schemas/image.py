#!/usr/bin/env python3
"""Image processing schemas for request/response handling."""

from enum import Enum
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, Field


class ImageType(str, Enum):
    """Enumeration of supported image analysis types."""
    DISEASE_DETECTION = "disease_detection"
    NUTRITION_ANALYSIS = "nutrition_analysis"
    PREGNANCY_DETECTION = "pregnancy_detection"
    PET_RECOGNITION = "pet_recognition"


class ImageValidation(BaseModel):
    """Schema for image validation requirements.

    Attributes:
        min_width: Minimum required width in pixels
        min_height: Minimum required height in pixels
        max_size_mb: Maximum allowed file size in MB
        allowed_formats: List of accepted image formats
    """
    min_width: int = Field(224, description="Minimum width in pixels")
    min_height: int = Field(224, description="Minimum height in pixels")
    max_size_mb: float = Field(5.0, description="Maximum file size in MB")
    allowed_formats: List[str] = Field(
        ["jpg", "jpeg", "png"],
        description="Allowed image formats"
    )


class ImageQuality(BaseModel):
    """Schema for image quality assessment.

    Attributes:
        brightness: Image brightness score
        contrast: Image contrast score
        sharpness: Image sharpness score
        is_valid: Quality validation flag
    """
    brightness: float = Field(..., description="Image brightness score (0-1)")
    contrast: float = Field(..., description="Image contrast score (0-1)")
    sharpness: float = Field(..., description="Image sharpness score (0-1)")
    is_valid: bool = Field(
        ...,
        description="Whether the image meets quality standards"
    )


class ImageUploadResponse(BaseModel):
    """Schema for image upload responses.

    Attributes:
        success: Upload success flag
        path: Storage path if successful
        error: Error message if failed
        url: Public URL if available
    """
    success: bool
    path: Optional[str] = None
    error: Optional[str] = None
    url: Optional[str] = None


class ImageMetadata(BaseModel):
    """Schema for image metadata storage.

    Attributes:
        filename: Original file name
        content_type: MIME type
        size_bytes: File size in bytes
        upload_date: Upload timestamp
        s3_path: Storage path in S3
        image_type: Type of analysis
    """
    filename: str
    content_type: str
    size_bytes: int
    upload_date: datetime
    s3_path: str
    image_type: ImageType
