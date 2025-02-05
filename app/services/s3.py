#!/usr/bin/env python3
"""S3 service for image storage and retrieval.

Provides functionality for managing images and their labels in AWS S3.
"""

from typing import Optional, BinaryIO, Tuple, List, Dict, Any
from datetime import datetime
import os
from io import BytesIO
import logging
import json

import boto3
from botocore.exceptions import ClientError
from PIL import Image

from app.core.config import settings
from app.schemas.image import ImageType
from app.core.logging import get_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3Service:
    """Service for managing image storage in AWS S3.

    Handles image upload, retrieval, and label management.
    """

    def __init__(self):
        """Initialize S3 client and configuration."""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            self.bucket_name = settings.S3_BUCKET
            self.labels_prefix = "labels/"
            self.images_prefix = "images/"
            logger.info(
                f"S3Service initialized with bucket: {self.bucket_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing S3Service: {str(e)}")
            raise

    def _generate_file_path(self, image_type: ImageType, filename: str) -> str:
        """Generate a unique file path for S3.

        Args:
            image_type: Type of image analysis
            filename: Original filename

        Returns:
            Generated S3 path
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{image_type.value}/{timestamp}_{filename}"

    def upload_image(
        self,
        image: Image.Image,
        filename: str,
        image_type: ImageType,
        content_type: str = "image/jpeg"
    ) -> Tuple[bool, str]:
        """Upload an image to S3.

        Args:
            image: PIL Image object
            filename: Original filename
            image_type: Type of image analysis
            content_type: MIME type of image

        Returns:
            Tuple of (success flag, path or error message)
        """
        try:
            logger.info(
                f"Attempting to upload image {filename} of type {image_type}"
            )
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format=image.format or 'JPEG')
            img_byte_arr.seek(0)

            s3_path = self._generate_file_path(image_type, filename)

            self.s3_client.upload_fileobj(
                img_byte_arr,
                self.bucket_name,
                s3_path,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'private'
                }
            )

            return True, s3_path
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return False, str(e)

    def get_image(self, s3_path: str) -> Optional[Image.Image]:
        """Retrieve an image from S3.

        Args:
            s3_path: S3 path to image

        Returns:
            PIL Image if found, None otherwise
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_path
            )
            image_data = response['Body'].read()
            return Image.open(BytesIO(image_data))
        except ClientError:
            return None

    def delete_image(self, s3_path: str) -> bool:
        """Delete an image from S3.

        Args:
            s3_path: S3 path to image

        Returns:
            True if deleted successfully
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_path
            )
            return True
        except ClientError:
            return False

    def get_image_url(
        self,
        s3_path: str,
        expires_in: int = 3600
    ) -> Optional[str]:
        """Generate a presigned URL for an image.

        Args:
            s3_path: S3 path to image
            expires_in: URL expiration time in seconds

        Returns:
            Presigned URL if successful
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_path
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError:
            return None

    def list_unlabeled_images(
        self,
        skip: int = 0,
        limit: int = 50
    ) -> List[str]:
        """List images that haven't been labeled yet.

        Args:
            skip: Number of images to skip
            limit: Maximum number of images to return

        Returns:
            List of unlabeled image paths
        """
        try:
            all_images = set()
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.images_prefix
            ):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key'].replace(self.images_prefix, '')
                        if key:
                            all_images.add(key)

            labeled_images = set()
            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.labels_prefix
            ):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key'].replace(
                            self.labels_prefix, ''
                        ).replace('.json', '')
                        if key:
                            labeled_images.add(key)

            unlabeled_images = list(all_images - labeled_images)
            unlabeled_images.sort()

            start = skip
            end = skip + limit
            return unlabeled_images[start:end]

        except Exception as e:
            logger.error(f"Error listing unlabeled images: {str(e)}")
            raise

    def get_labels(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get labels for a specific image.

        Args:
            image_id: Image identifier

        Returns:
            Label data if found
        """
        try:
            key = f"{self.labels_prefix}{image_id}.json"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.error(f"Error getting labels for {image_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting labels for {image_id}: {str(e)}")
            raise

    def save_labels(self, image_id: str, labels: Dict[str, Any]) -> bool:
        """Save or update labels for a specific image.

        Args:
            image_id: Image identifier
            labels: Label data to save

        Returns:
            True if saved successfully
        """
        try:
            key = f"{self.labels_prefix}{image_id}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(labels),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            logger.error(f"Error saving labels for {image_id}: {str(e)}")
            return False

    def image_exists(self, image_id: str) -> bool:
        """Check if an image exists in S3.

        Args:
            image_id: Image identifier

        Returns:
            True if image exists
        """
        try:
            key = f"{self.images_prefix}{image_id}"
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking image existence: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error checking image existence: {str(e)}")
            raise

    def count_total_images(self) -> int:
        """Count total number of images.

        Returns:
            Total number of images
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.images_prefix
            )
            return response.get('KeyCount', 0) - 1
        except Exception as e:
            logger.error(f"Error counting total images: {str(e)}")
            raise

    def count_labeled_images(self) -> int:
        """Count number of labeled images.

        Returns:
            Number of labeled images
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.labels_prefix
            )
            return response.get('KeyCount', 0) - 1
        except Exception as e:
            logger.error(f"Error counting labeled images: {str(e)}")
            raise
