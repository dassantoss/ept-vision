#!/usr/bin/env python3
"""
Dataset management endpoints for the API.

This module handles operations related to dataset management including
image upload, labeling, export, and pre-labeling functionality.
"""

from typing import List, Dict, Any, Optional
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Request,
    Query,
    Path
)
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks
import os
import io
from PIL import Image

from app.core.database import get_db
from app.services.s3 import S3Service
from app.core.logging import get_logger
from app.tools.dataset_manager import DatasetManager
from app.models.prelabeling.model import PreLabelingModel

logger = get_logger("ept_vision.dataset_api")
router = APIRouter()
s3_service = S3Service()
dataset_manager = DatasetManager()
prelabeling_model = PreLabelingModel()


class ImageLabel(BaseModel):
    """Model for image labeling data."""
    animal_type: str
    size: str
    body_condition: str
    visible_health_issues: str
    pregnancy_indicators: str
    image_quality: str
    context: str


@router.post("/upload", response_model=Dict[str, str])
async def upload_image(
    file: UploadFile = File(...),
    request: Request = None
) -> Dict[str, str]:
    """
    Upload an image to the dataset.

    Args:
        file: Image file to upload
        request: FastAPI request object

    Returns:
        Dict with filename of uploaded image

    Raises:
        HTTPException: If upload fails
    """
    try:
        filename = await dataset_manager.save_image(file)
        logger.info(f"Image uploaded successfully: {filename}")
        return {"filename": filename}
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error uploading image: {str(e)}"
        )


@router.get("/unlabeled", response_model=Dict[str, List[str]])
async def get_unlabeled_images(
    request: Request,
    skip: int = 0,
    limit: int = 50
) -> Dict[str, List[str]]:
    """
    Get a batch of unlabeled images.

    Args:
        request: FastAPI request object
        skip: Number of images to skip
        limit: Maximum number of images to return

    Returns:
        Dict containing list of unlabeled image IDs

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        images = dataset_manager.get_unlabeled_images(skip=skip, limit=limit)
        logger.info(f"Retrieved {len(images)} unlabeled images")
        return {"images": images}
    except Exception as e:
        logger.error(f"Error getting unlabeled images: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/labels/{image_id}", response_model=Dict[str, Any])
async def update_labels(
    image_id: str,
    labels: ImageLabel,
    request: Request
) -> Dict[str, Any]:
    """
    Update labels for a specific image.

    Args:
        image_id: ID of the image to update
        labels: New labels to apply
        request: FastAPI request object

    Returns:
        Dict confirming update success

    Raises:
        HTTPException: If update fails
    """
    try:
        existing_labels = dataset_manager.get_labels(image_id) or {}
        new_labels = labels.dict()
        existing_labels.update(new_labels)
        dataset_manager.update_labels(image_id, existing_labels)
        logger.info(f"Labels updated for image: {image_id}")
        return {"message": "Labels updated successfully"}
    except Exception as e:
        logger.error(f"Error updating labels: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@router.get("/labels/{image_id}", response_model=Dict[str, Any])
async def get_image_labels(
    image_id: str,
    request: Request
) -> Dict[str, Any]:
    """
    Get labels for a specific image.

    Args:
        image_id: ID of the image to get labels for
        request: FastAPI request object

    Returns:
        Dict containing image labels

    Raises:
        HTTPException: If labels not found or retrieval fails
    """
    try:
        labels = dataset_manager.get_labels(image_id)
        if labels is None:
            raise HTTPException(
                status_code=404,
                detail="Labels not found for this image"
            )
        return labels
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting labels: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/progress", response_model=Dict[str, Any])
async def get_labeling_progress(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get the overall progress of image labeling.

    Args:
        db: Database session

    Returns:
        Dict containing labeling progress statistics

    Raises:
        HTTPException: If progress calculation fails
    """
    try:
        total_images = s3_service.count_total_images()
        labeled_images = s3_service.count_labeled_images()
        progress = \
            (labeled_images / total_images * 100) if total_images > 0 else 0

        return {
            "total_images": total_images,
            "labeled_images": labeled_images,
            "progress_percentage": progress
        }
    except Exception as e:
        logger.error(f"Error getting labeling progress: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_dataset_statistics(request: Request) -> Dict[str, Any]:
    """
    Get dataset statistics including label distributions and progress.

    Args:
        request: FastAPI request object

    Returns:
        Dict containing dataset statistics

    Raises:
        HTTPException: If statistics calculation fails
    """
    try:
        stats = dataset_manager.get_dataset_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting dataset statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting dataset statistics: {str(e)}"
        )


@router.get("/export/{format}")
async def export_dataset(
    format: str,
    request: Request,
    background_tasks: BackgroundTasks
) -> FileResponse:
    """
    Export dataset in specified format.

    Args:
        format: Export format ('json' or 'csv')
        request: FastAPI request object
        background_tasks: Background tasks handler

    Returns:
        FileResponse with exported dataset

    Raises:
        HTTPException: If export fails or format is invalid
    """
    try:
        if format.lower() not in ['json', 'csv']:
            raise ValueError(f"Unsupported format: {format}")

        output_file = dataset_manager.export_dataset(format)

        return FileResponse(
            output_file,
            filename=f"dataset_export.{format.lower()}",
            media_type='application/octet-stream'
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting dataset: {str(e)}"
        )


@router.get("/training-data/{model_type}")
async def get_training_data(
    model_type: str,
    request: Request
) -> Dict[str, Any]:
    """
    Get dataset formatted for specific model training.

    Args:
        model_type: Type of model to format data for
        request: FastAPI request object

    Returns:
        Dict containing formatted training data

    Raises:
        HTTPException: If data formatting fails
    """
    try:
        training_data = dataset_manager.get_model_training_data(model_type)
        return training_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting training data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting training data: {str(e)}"
        )


@router.get("/images", response_model=Dict[str, Any])
async def get_images(
    request: Request,
    skip: int = 0,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get paginated list of images.

    Args:
        request: FastAPI request object
        skip: Number of images to skip
        limit: Maximum number of images per page

    Returns:
        Dict containing paginated image list and metadata

    Raises:
        HTTPException: If image listing fails
    """
    try:
        all_images = []
        for f in os.listdir(dataset_manager.raw_dir):
            if os.path.splitext(f)[1].lower() in dataset_manager.VALID_EXTENSIONS:
                all_images.append(f)

        all_images.sort()
        total_images = len(all_images)
        paginated_images = all_images[skip:skip + limit]

        logger.info(f"Found {len(paginated_images)} images")
        return {
            "images": paginated_images,
            "total": total_images,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/image/{image_id}")
async def get_image(
    image_id: str,
    request: Request
) -> FileResponse:
    """
    Get an image file by its ID.

    Args:
        image_id: ID of the image to retrieve
        request: FastAPI request object

    Returns:
        FileResponse containing the image

    Raises:
        HTTPException: If image not found or retrieval fails
    """
    try:
        image_path = os.path.join(dataset_manager.raw_dir, image_id)

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )

        media_type = dataset_manager._get_media_type(image_path)

        return FileResponse(
            image_path,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/debug/directory")
async def debug_directory(request: Request) -> Dict[str, Any]:
    """
    Debug endpoint to check directory contents.

    Args:
        request: FastAPI request object

    Returns:
        Dict containing directory information

    Raises:
        HTTPException: If directory check fails
    """
    try:
        raw_dir = dataset_manager.raw_dir
        valid_images = []

        if os.path.exists(raw_dir) and os.path.isdir(raw_dir):
            valid_images = [
                f for f in os.listdir(raw_dir)
                if os.path.isfile(os.path.join(raw_dir, f)) and
                os.path.splitext(f)[1].lower() in dataset_manager.VALID_EXTENSIONS
            ]

        return {
            'raw_dir': raw_dir,
            'exists': os.path.exists(raw_dir),
            'is_dir': os.path.isdir(raw_dir) if os.path.exists(raw_dir) else False,
            'files': os.listdir(raw_dir) if os.path.exists(raw_dir) and os.path.isdir(raw_dir) else [],
            'valid_images': valid_images
        }
    except Exception as e:
        logger.error(f"Error debugging directory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error debugging directory: {str(e)}"
        )


@router.post("/discard/{image_id}")
async def discard_image(
    image_id: str = Path(..., description="ID of the image to discard")
) -> Dict[str, Any]:
    """
    Discard an image from the dataset.

    Args:
        image_id: ID of the image to discard

    Returns:
        Dict containing discard operation result

    Raises:
        HTTPException: If discard operation fails
    """
    try:
        logger.info(f"Attempting to discard image {image_id}")

        os.makedirs(dataset_manager.raw_dir, exist_ok=True)
        os.makedirs(os.path.dirname(dataset_manager.discarded_file),
                    exist_ok=True)

        image_path = os.path.join(dataset_manager.raw_dir, image_id)
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Image {image_id} not found"
            )

        success = dataset_manager.discard_image(
            image_id,
            reason="manual_discard"
        )

        if not success:
            logger.error(f"Failed to discard image {image_id}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to discard image {image_id}"
            )

        logger.info(f"Successfully discarded image {image_id}")
        return {
            "success": True,
            "message": f"Image {image_id} discarded successfully",
            "image_id": image_id
        }
  
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error discarding image {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/prelabel", response_model=Dict[str, Any])
async def prelabel_image(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Pre-label an image using the automatic labeling model.

    Args:
        file: Image file to pre-label

    Returns:
        Dict containing pre-labeling results

    Raises:
        HTTPException: If pre-labeling fails
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        predictions = prelabeling_model(image)

        logger.info(f"Pre-labeling results for {file.filename}: {predictions}")
        return predictions

    except Exception as e:
        logger.error(f"Error pre-labeling image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error pre-labeling image: {str(e)}"
        )


@router.post("/prelabel/batch", response_model=Dict[str, Any])
async def prelabel_dataset() -> Dict[str, Any]:
    """
    Pre-label all unlabeled images in the dataset.

    Returns:
        Dict containing batch pre-labeling statistics

    Raises:
        HTTPException: If batch pre-labeling fails
    """
    try:
        stats = {
            "processed": 0,
            "labeled": 0,
            "discarded": 0,
            "errors": 0
        }

        unlabeled = dataset_manager.get_unlabeled_images()
        stats["processed"] = len(unlabeled)
        
        for image_id in unlabeled:
            try:
                image_path = os.path.join(dataset_manager.raw_dir, image_id)
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_id}")
                    stats["errors"] += 1
                    continue

                image = Image.open(image_path)
                predictions = prelabeling_model(image)

                if predictions["should_discard"]:
                    dataset_manager.discard_image(image_id, "auto_discarded")
                    stats["discarded"] += 1
                    logger.info(f"Auto-discarded image: {image_id}")
                else:
                    dataset_manager.update_labels(
                        image_id,
                        predictions["labels"]
                    )
                    stats["labeled"] += 1
                    logger.info(f"Auto-labeled image: {image_id}")

            except Exception as e:
                logger.error(f"Error processing image {image_id}: {str(e)}")
                stats["errors"] += 1
                continue

        return stats

    except Exception as e:
        logger.error(f"Error in batch pre-labeling: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch pre-labeling: {str(e)}"
        )


@router.post("/prelabel/{image_id}")
async def prelabel_single_image(
    image_id: str = Path(..., description="ID of the image to prelabel")
) -> Dict[str, Any]:
    """
    Pre-label a single image using the model.

    Args:
        image_id: ID of the image to pre-label

    Returns:
        Dict containing pre-labeling results

    Raises:
        HTTPException: If pre-labeling fails
    """
    try:
        logger.info(f"Starting pre-labeling for image {image_id}")

        image_path = os.path.join(dataset_manager.raw_dir, image_id)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            raise HTTPException(status_code=404, detail="Image file not found")

        try:
            with Image.open(image_path) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                predictions = prelabeling_model(image)

                if not predictions["should_discard"]:
                    dataset_manager.update_labels(image_id, predictions["labels"])
                    logger.info(f"Labels saved for image {image_id}")
                else:
                    dataset_manager.discard_image(image_id, "auto_discarded")
                    logger.info(f"Image {image_id} marked as discarded")

                logger.info(f"Pre-labeling predictions for {image_id}: {predictions}")
                return predictions

        except (IOError, OSError) as e:
            logger.error(f"Error reading image file {image_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading image file: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error pre-labeling image {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error pre-labeling image: {str(e)}"
        )
