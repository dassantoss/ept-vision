#!/usr/bin/env python3
"""
Pre-labeling endpoints for the API.

This module provides endpoints for automatic image pre-labeling using
machine learning models, including batch processing and streaming capabilities.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Request
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import json
import asyncio
from datetime import datetime

from app.models.prelabeling.model import PreLabelingModel
from app.core.logging import get_logger
from app.tools.dataset_manager import DatasetManager

router = APIRouter()
logger = get_logger("ept_vision.api.prelabel")

# Initialize models
try:
    prelabeling_model = PreLabelingModel()
    logger.info("Pre-labeling model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing pre-labeling model: {str(e)}")
    raise


def format_sse_event(data: Dict[str, Any]) -> str:
    """
    Format data as Server-Sent Event.

    Args:
        data: Dictionary containing event data

    Returns:
        str: Formatted SSE event string
    """
    json_data = json.dumps(data)
    return f"data: {json_data}\n\n"


async def process_image(
    dataset_manager: DatasetManager,
    image_id: str,
    idx: int
) -> Dict[str, Any]:
    """
    Process a single image and return its result.

    Args:
        dataset_manager: Instance of DatasetManager
        image_id: ID of the image to process
        idx: Index of the image in the batch

    Returns:
        Dict containing processing results
    """
    try:
        if dataset_manager.is_image_discarded(image_id):
            logger.info(f"Image {image_id} was already discarded, skipping")
            return {
                "type": "progress",
                "index": idx,
                "image_id": image_id,
                "status": "skipped",
                "reason": "already_discarded"
            }

        existing_labels = dataset_manager.get_labels(image_id)
        if existing_labels:
            logger.info(f"Image {image_id} already has labels, skipping")
            return {
                "type": "progress",
                "index": idx,
                "image_id": image_id,
                "status": "skipped",
                "reason": "already_labeled",
                "labels": existing_labels
            }

        image = dataset_manager.get_image(image_id)
        if image is None:
            logger.error(f"Could not load image {image_id}")
            return {
                "type": "progress",
                "index": idx,
                "image_id": image_id,
                "status": "error",
                "reason": "image_not_found"
            }

        logger.info(f"Getting predictions for {image_id}")
        predictions = prelabeling_model(image)

        if predictions.get("should_discard", False):
            reason = predictions.get(
                "reason",
                "Low confidence or no animal detected"
            )
            logger.info(f"Model suggests discarding image {image_id}: {reason}")

            dataset_manager.discard_image(image_id, reason=reason)
            return {
                "type": "progress",
                "index": idx,
                "image_id": image_id,
                "status": "discarded",
                "reason": reason,
                "confidence": predictions.get("confidence", 0.0)
            }

        if "labels" in predictions:
            logger.info(
                f"Updating labels for {image_id}: {predictions['labels']}"
            )
            dataset_manager.update_labels(image_id, predictions["labels"])
            return {
                "type": "progress",
                "index": idx,
                "image_id": image_id,
                "status": "labeled",
                "labels": predictions["labels"],
                "confidence": predictions.get("confidence", 0.0)
            }

        logger.warning(f"No labels in predictions for {image_id}")
        return {
            "type": "progress",
            "index": idx,
            "image_id": image_id,
            "status": "error",
            "reason": "no_labels_generated"
        }

    except Exception as e:
        logger.error(f"Error processing image {image_id}: {str(e)}")
        return {
            "type": "progress",
            "index": idx,
            "image_id": image_id,
            "status": "error",
            "reason": str(e)
        }


@router.get("/dataset/prelabel/batch/stream")
async def stream_prelabel_dataset(
    images: str = Query(..., description="JSON string of image IDs to process"),
    request: Request = None
) -> StreamingResponse:
    """
    Stream pre-labeling progress for specific images.

    Args:
        images: JSON string containing list of image IDs
        request: FastAPI request object

    Returns:
        StreamingResponse: Server-sent events stream

    Raises:
        HTTPException: If image list format is invalid
    """
    try:
        image_list = json.loads(images)
        return StreamingResponse(
            stream_prelabel_generator(image_list, request),
            media_type="text/event-stream"
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400,
                            detail="Invalid image list format")


async def stream_prelabel_generator(
    image_list: list,
    request: Request = None
):
    """
    Generator for streaming pre-labeling progress.

    Args:
        image_list: List of images to process
        request: FastAPI request object

    Yields:
        str: Formatted SSE event string
    """
    try:
        dataset_manager = DatasetManager()
        logger.info(
            f"Starting batch pre-labeling stream for {len(image_list)} images"
        )

        for idx, image_data in enumerate(image_list):
            if request and await request.is_disconnected():
                logger.info("Client disconnected, stopping batch processing")
                yield format_sse_event({
                    "type": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "status": "canceled"
                })
                return

            image_id = image_data['id']
            result = await process_image(dataset_manager, image_id, idx)
            yield format_sse_event(result)
            await asyncio.sleep(0.1)

        yield format_sse_event({
            "type": "complete",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })

    except Exception as e:
        logger.error(f"Error in batch pre-labeling stream: {str(e)}")
        yield format_sse_event({
            "type": "error",
            "error": str(e)
        })


@router.post("/dataset/prelabel/{image_id}")
async def prelabel_image(image_id: str) -> Dict[str, Any]:
    """
    Pre-label a single image using our pre-labeling model.

    Args:
        image_id: ID of the image to pre-label

    Returns:
        Dict containing pre-labeling results

    Raises:
        HTTPException: If pre-labeling fails
    """
    try:
        dataset_manager = DatasetManager()
        logger.info(f"Processing image {image_id}")

        if dataset_manager.is_image_discarded(image_id):
            logger.info(f"Image {image_id} was previously discarded")
            return {
                "should_discard": True,
                "reason": "Image was previously discarded",
                "confidence": 0.0
            }

        image = dataset_manager.get_image(image_id)
        if image is None:
            logger.error(f"Image {image_id} not found")
            raise HTTPException(status_code=404, detail="Image not found")

        logger.info(f"Getting predictions for image {image_id}")
        predictions = prelabeling_model(image)
        logger.info(f"Raw predictions for {image_id}: {predictions}")

        if predictions.get("should_discard", False):
            reason = predictions.get(
                "reason",
                "Low confidence or no animal detected"
            )
            logger.info(f"Model suggests discarding image {image_id}: {reason}")

            try:
                dataset_manager.discard_image(
                    image_id,
                    reason=reason
                )
                logger.info(
                    f"Successfully marked image {image_id} as discarded"
                )

                return {
                    "should_discard": True,
                    "reason": reason,
                    "confidence": predictions.get("confidence", 0.0)
                }
            except Exception as e:
                logger.error(f"Error discarding image {image_id}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error discarding image: {str(e)}"
                )

        if "labels" in predictions:
            try:
                dataset_manager.update_labels(image_id, predictions["labels"])
                logger.info(
                    f"Updated labels for {image_id}: {predictions['labels']}"
                )
            except Exception as e:
                logger.error(
                    f"Error updating labels for {image_id}: {str(e)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Error updating labels: {str(e)}"
                )

        result = {
            "should_discard": False,
            "labeled": True,
            "confidence": predictions.get("confidence", 0.0),
            "labels": predictions.get("labels", {}),
            "animal_type": predictions.get("labels", {}).get(
                "animal_type",
                "unknown"
            )
        }

        logger.info(
            f"Successfully pre-labeled image {image_id} with result: {result}"
        )
        return result

    except Exception as e:
        logger.error(f"Error pre-labeling image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/prelabel/batch")
async def prelabel_dataset(
    start: int = Query(0, description="Starting index for pagination"),
    limit: int = Query(100, description="Number of images to process")
) -> Dict[str, Any]:
    """
    Pre-label a batch of unlabeled images in the dataset.

    Args:
        start: Starting index for pagination
        limit: Number of images to process

    Returns:
        Dict containing batch processing results

    Raises:
        HTTPException: If batch processing fails
    """
    try:
        dataset_manager = DatasetManager()
        logger.info(
            f"Starting batch pre-labeling with start={start}, limit={limit}"
        )

        unlabeled_images = dataset_manager.get_unlabeled_images(
            skip=start,
            limit=limit
        )
        logger.info(
            f"Found {len(unlabeled_images)} unlabeled images in current page"
        )

        labeled_count = 0
        discarded_count = 0
        error_count = 0
        processed_count = 0
        processed_details = []

        for idx, image_id in enumerate(unlabeled_images):
            try:
                logger.info(
                    f"Processing image {idx + 1}/{len(unlabeled_images)}: "
                    f"{image_id}"
                )
                processed_count += 1

                if dataset_manager.is_image_discarded(image_id):
                    logger.info(
                        f"Image {image_id} was already discarded, skipping"
                    )
                    processed_details.append({
                        "image_id": image_id,
                        "status": "skipped",
                        "reason": "already_discarded"
                    })
                    continue

                image = dataset_manager.get_image(image_id)
                if image is None:
                    logger.error(f"Could not load image {image_id}")
                    error_count += 1
                    processed_details.append({
                        "image_id": image_id,
                        "status": "error",
                        "reason": "image_not_found"
                    })
                    continue

                logger.info(f"Getting predictions for {image_id}")
                predictions = prelabeling_model(image)

                if predictions.get("should_discard", False):
                    reason = predictions.get(
                        "reason",
                        "Low confidence or no animal detected"
                    )
                    logger.info(
                        f"Model suggests discarding image {image_id}: {reason}"
                    )

                    dataset_manager.discard_image(image_id, reason=reason)
                    discarded_count += 1
                    processed_details.append({
                        "image_id": image_id,
                        "status": "discarded",
                        "reason": reason,
                        "confidence": predictions.get("confidence", 0.0)
                    })
                    continue

                if "labels" in predictions:
                    logger.info(
                        f"Updating labels for {image_id}: {predictions['labels']}"
                    )
                    dataset_manager.update_labels(image_id, predictions["labels"])
                    labeled_count += 1
                    processed_details.append({
                        "image_id": image_id,
                        "status": "labeled",
                        "labels": predictions["labels"],
                        "confidence": predictions.get("confidence", 0.0)
                    })
                else:
                    logger.warning(f"No labels in predictions for {image_id}")
                    processed_details.append({
                        "image_id": image_id,
                        "status": "error",
                        "reason": "no_labels_generated"
                    })

            except Exception as e:
                logger.error(f"Error processing image {image_id}: {str(e)}")
                error_count += 1
                processed_details.append({
                    "image_id": image_id,
                    "status": "error",
                    "reason": str(e)
                })
                continue

        logger.info(
            "Batch processing complete. "
            f"Results: labeled={labeled_count}, "
            f"discarded={discarded_count}, "
            f"errors={error_count}, "
            f"total_processed={processed_count}"
        )

        return {
            "success": True,
            "labeled": labeled_count,
            "discarded": discarded_count,
            "errors": error_count,
            "total_processed": processed_count,
            "processed_details": processed_details
        }

    except Exception as e:
        logger.error(f"Error in batch pre-labeling: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset/prelabel/{image_id}")
async def preview_prelabel_image(image_id: str) -> Dict[str, Any]:
    """
    Preview pre-label results for a single image without saving.

    Args:
        image_id: ID of the image to preview

    Returns:
        Dict containing preview results

    Raises:
        HTTPException: If preview fails
    """
    try:
        dataset_manager = DatasetManager()
        logger.info(f"Processing preview for image {image_id}")

        image = dataset_manager.get_image(image_id)
        if image is None:
            logger.error(f"Image {image_id} not found")
            raise HTTPException(status_code=404, detail="Image not found")

        logger.info(f"Getting preview predictions for image {image_id}")
        predictions = prelabeling_model(image)

        return {
            "should_discard": predictions.get("should_discard", False),
            "confidence": predictions.get("confidence", 0.0),
            "labels": predictions.get("labels", {}),
            "size_confidence": predictions.get("size_confidence", 0.0),
            "health_confidence": predictions.get("health_confidence", 0.0),
            "pregnancy_confidence": predictions.get(
                "pregnancy_confidence",
                0.0
            ),
            "body_condition_confidence": predictions.get(
                "body_condition_confidence",
                0.0
            ),
            "quality_confidence": predictions.get("quality_confidence", 0.0),
            "context_confidence": predictions.get("context_confidence", 0.0)
        }

    except Exception as e:
        logger.error(
            f"Error in preview pre-labeling for image {image_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
