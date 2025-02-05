#!/usr/bin/env python3
"""Dataset management module for image processing.

This module provides functionality for managing image datasets, including:
- Image storage and retrieval
- Label management
- Dataset statistics
- Export capabilities
"""

import os
import json
import uuid
from typing import List, Dict, Any, Set
from PIL import Image
import io
from fastapi import UploadFile
from datetime import datetime
import mimetypes
from app.core.logging import get_logger
import shutil
import time

logger = get_logger("ept_vision.dataset_manager")


class DatasetManager:
    """Manager for image dataset operations.

    Handles storage, retrieval, and management of images and their labels.
    Provides functionality for dataset organization and export.

    Attributes:
        VALID_EXTENSIONS: Set of allowed image file extensions
        data_dir: Base directory for data storage
        dataset_dir: Directory for dataset files
        raw_dir: Directory for original images
        processed_dir: Directory for processed images
        labels_file: Path to labels JSON file
        discarded_file: Path to discarded images JSON file
        counter_file: Path to image counter file
    """

    VALID_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png'}

    def __init__(self):
        """Initialize the dataset manager."""
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.dataset_dir = os.path.join(self.data_dir, "datasets")
        self.raw_dir = os.path.join(self.dataset_dir, "raw_images")
        self.processed_dir = os.path.join(self.dataset_dir, "processed_images")
        self.labels_file = os.path.join(self.dataset_dir, "labels.json")
        self.discarded_file = os.path.join(self.dataset_dir, "discarded_images.json")
        self.counter_file = os.path.join(self.dataset_dir, "image_counter.txt")

        # Ensure directories exist
        for directory in [self.dataset_dir, self.raw_dir, self.processed_dir]:
            os.makedirs(directory, exist_ok=True)

        # Initialize files if they don't exist
        self._initialize_files()

    def _initialize_files(self):
        """Initialize necessary files if they don't exist."""
        try:
            # Initialize labels file
            if not os.path.exists(self.labels_file):
                with open(self.labels_file, 'w') as f:
                    json.dump({}, f, indent=2)
                logger.info("Created empty labels file")

            # Initialize discarded images file
            if not os.path.exists(self.discarded_file):
                with open(self.discarded_file, 'w') as f:
                    json.dump({
                        "metadata": {
                            "created_at": datetime.now().isoformat(),
                            "last_updated": datetime.now().isoformat(),
                            "total_discarded": 0
                        },
                        "discarded_images": {}
                    }, f, indent=2)
                logger.info("Created empty discarded images file")

            # Initialize counter file
            if not os.path.exists(self.counter_file):
                with open(self.counter_file, 'w') as f:
                    f.write('0')
                logger.info("Created counter file starting at 0")
                
        except Exception as e:
            logger.error(f"Error initializing files: {str(e)}")
            raise

    def _load_discarded_images(self) -> Dict[str, Any]:
        """Load discarded images from file.

        Returns:
            Dictionary containing discarded images data
        """
        try:
            with open(self.discarded_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading discarded images: {str(e)}")
            return {"metadata": {}, "discarded_images": {}}

    def _save_discarded_images(self, data: Dict[str, Any]) -> None:
        """Save discarded images to file.

        Args:
            data: Dictionary containing discarded images data
        """
        try:
            data["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.discarded_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving discarded images: {str(e)}")
            raise

    def discard_image(self, image_id: str, reason: str = "not_animal") -> bool:
        """Mark an image as discarded.

        Args:
            image_id: Identifier of the image to discard
            reason: Reason for discarding the image

        Returns:
            True if image was successfully discarded
        """
        try:
            logger.info(
                f"Attempting to discard image {image_id} "
                f"with reason: {reason}"
            )

            image_path = os.path.join(self.raw_dir, image_id)
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_id}")
                return False

            if not os.path.exists(self.discarded_file):
                self._initialize_files()

            try:
                with open(self.discarded_file, 'r') as f:
                    discarded_data = json.load(f)
            except Exception as e:
                logger.error(
                    f"Error loading discarded images data: {str(e)}"
                )
                discarded_data = {
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "total_discarded": 0
                    },
                    "discarded_images": {}
                }

            if image_id in discarded_data.get("discarded_images", {}):
                logger.info(f"Image {image_id} is already discarded")
                return True

            discarded_data["discarded_images"][image_id] = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason
            }

            discarded_data["metadata"]["last_updated"] = (
                datetime.now().isoformat()
            )
            discarded_data["metadata"]["total_discarded"] = len(
                discarded_data["discarded_images"]
            )

            try:
                with open(self.discarded_file, 'w') as f:
                    json.dump(discarded_data, f, indent=2)
                logger.info(f"Updated discarded_images.json for {image_id}")
            except Exception as e:
                logger.error(
                    f"Error saving discarded images data: {str(e)}"
                )
                return False

            try:
                if os.path.exists(self.labels_file):
                    with open(self.labels_file, 'r') as f:
                        labels = json.load(f)
                    if image_id in labels:
                        del labels[image_id]
                        with open(self.labels_file, 'w') as f:
                            json.dump(labels, f, indent=2)
                        logger.info(f"Removed {image_id} from labels.json")
            except Exception as e:
                logger.warning(
                    f"Error removing discarded image from labels: {str(e)}"
                )

            logger.info(f"Image {image_id} discarded successfully")
            return True

        except Exception as e:
            logger.error(f"Error discarding image {image_id}: {str(e)}")
            return False

    def is_image_discarded(self, image_id: str) -> bool:
        """Check if an image is marked as discarded.

        Args:
            image_id: Identifier of the image to check

        Returns:
            True if image is marked as discarded
        """
        try:
            discarded_data = self._load_discarded_images()
            return image_id in discarded_data.get("discarded_images", {})
        except Exception as e:
            logger.error(
                f"Error checking if image is discarded: {str(e)}"
            )
            return False

    def get_unlabeled_images(
        self,
        skip: int = 0,
        limit: int = 50
    ) -> List[str]:
        """Get a list of unlabeled images, excluding discarded ones.

        Args:
            skip: Number of images to skip for pagination
            limit: Maximum number of images to return

        Returns:
            List of unlabeled image identifiers
        """
        try:
            all_images = self.get_all_valid_images()
            logger.info(f"Total valid images: {len(all_images)}")

            with open(self.labels_file, 'r') as f:
                all_labels = json.load(f)
                labels = {
                    img_id: label_data
                    for img_id, label_data in all_labels.items()
                    if label_data and any(label_data.values())
                }

            discarded_data = self._load_discarded_images()
            discarded_images = discarded_data.get("discarded_images", {})

            unlabeled = sorted([
                img for img in all_images
                if img not in labels and img not in discarded_images
            ])

            logger.info(f"Found {len(unlabeled)} unlabeled images")
            logger.info(f"Applying pagination: skip={skip}, limit={limit}")

            start_idx = min(skip, len(unlabeled))
            end_idx = min(start_idx + limit, len(unlabeled))
            paginated_unlabeled = unlabeled[start_idx:end_idx]

            logger.info(
                f"Returning {len(paginated_unlabeled)} images "
                f"(from index {start_idx} to {end_idx})"
            )
            return paginated_unlabeled

        except Exception as e:
            logger.error(f"Error getting unlabeled images: {str(e)}")
            return []

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics, excluding discarded images.

        Returns:
            Dictionary containing dataset statistics and distributions
        """
        try:
            all_images = self.get_all_valid_images()

            with open(self.labels_file, 'r') as f:
                labels = json.load(f)

            discarded_data = self._load_discarded_images()
            discarded_images = discarded_data.get("discarded_images", {})

            total_images = len(all_images)
            discarded_count = len(discarded_images)
            valid_images = total_images - discarded_count
            labeled_images = len(labels)
            unlabeled_images = valid_images - labeled_images

            distributions = {
                'animal_type': {},
                'size': {},
                'body_condition': {},
                'visible_health_issues': {},
                'pregnancy_indicators': {},
                'image_quality': {},
                'context': {}
            }

            for label_data in labels.values():
                for category in distributions.keys():
                    if category in label_data:
                        value = label_data[category]
                        distributions[category][value] = (
                            distributions[category].get(value, 0) + 1
                        )

            return {
                'total_images': valid_images,
                'labeled_images': labeled_images,
                'unlabeled_images': unlabeled_images,
                'discarded_images': discarded_count,
                'progress_percentage': (
                    labeled_images / valid_images * 100
                    if valid_images > 0 else 0
                ),
                'distributions': distributions
            }

        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
            raise

    def _validate_image_file(self, file: UploadFile) -> None:
        """Validate image file type and format.

        Args:
            file: FastAPI UploadFile object to validate

        Raises:
            ValueError: If file extension or type is invalid
        """
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in self.VALID_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension. "
                f"Allowed: {', '.join(self.VALID_EXTENSIONS)}"
            )

        if not file.content_type.startswith('image/'):
            raise ValueError("File must be an image")

    def _get_media_type(self, filepath: str) -> str:
        """Get the media type for a file.

        Args:
            filepath: Path to the file

        Returns:
            MIME type string for the file
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.jpg' or ext == '.jpeg':
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        return 'application/octet-stream'

    def _get_next_counter(self) -> int:
        """Get and increment the image counter in a thread-safe way.

        Returns:
            Next counter value

        Raises:
            ValueError: If counter file operations fail
        """
        lock_file = f"{self.counter_file}.lock"

        try:
            while os.path.exists(lock_file):
                logger.debug("Lock file exists, waiting...")
                time.sleep(0.1)

            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))

            try:
                if not os.path.exists(self.counter_file):
                    logger.info("Counter file does not exist, creating it")
                    with open(self.counter_file, 'w') as f:
                        f.write('0')
                    return 0

                logger.info("Reading current counter value")
                with open(self.counter_file, 'r') as f:
                    content = f.read().strip()
                    logger.info(f"Read counter value: {content}")
                    counter = int(content)

                counter += 1
                logger.info(f"Incrementing counter to: {counter}")

                with open(self.counter_file, 'w', encoding='utf-8') as f:
                    f.write(str(counter))
                    f.flush()
                    os.fsync(f.fileno())

                logger.info(f"Successfully saved new counter value: {counter}")
                return counter

            finally:
                if os.path.exists(lock_file):
                    os.remove(lock_file)

        except Exception as e:
            logger.error(f"Error managing counter: {str(e)}")
            logger.error(f"Counter file path: {self.counter_file}")
            logger.error(
                f"Counter file exists: {os.path.exists(self.counter_file)}"
            )
            if os.path.exists(self.counter_file):
                logger.error(
                    f"Counter file permissions: "
                    f"{oct(os.stat(self.counter_file).st_mode)}"
                )
            if os.path.exists(lock_file):
                os.remove(lock_file)
            raise ValueError(f"Failed to get next counter: {str(e)}")

    async def save_image(self, file: UploadFile) -> str:
        """Save an uploaded image to the dataset.

        Args:
            file: FastAPI UploadFile object containing the image

        Returns:
            Generated filename for the saved image

        Raises:
            ValueError: If image validation or saving fails
        """
        try:
            self._validate_image_file(file)

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            counter = self._get_next_counter()
            suffix = f"{counter:06d}"

            extension = os.path.splitext(file.filename)[1].lower()

            filename = f"{timestamp}_{suffix}{extension}"
            filepath = os.path.join(self.raw_dir, filename)

            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image.verify()

            image = Image.open(io.BytesIO(contents))
            image.save(filepath)
            logger.info(f"Image saved successfully: {filename}")

            return filename

        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise

    def get_all_valid_images(self) -> List[str]:
        """Get all valid images from the raw directory.

        Returns:
            List of valid image filenames
        """
        try:
            if not os.path.exists(self.raw_dir) or not os.path.isdir(
                self.raw_dir
            ):
                logger.error(f"Raw directory is invalid: {self.raw_dir}")
                return []

            valid_images = []
            for filename in os.listdir(self.raw_dir):
                filepath = os.path.join(self.raw_dir, filename)
                if not os.path.isfile(filepath):
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.VALID_EXTENSIONS:
                    continue

                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_images.append(filename)
                except Exception as e:
                    logger.warning(
                        f"Invalid image file {filename}: {str(e)}"
                    )
                    continue

            return valid_images
        except Exception as e:
            logger.error(f"Error getting valid images: {str(e)}")
            return []

    def update_labels(self, image_id: str, labels: Dict[str, Any]) -> None:
        """Update labels for an image.

        Args:
            image_id: Identifier of the image to update
            labels: Dictionary containing label data

        Raises:
            ValueError: If image doesn't exist or label update fails
        """
        try:
            if not os.path.exists(os.path.join(self.raw_dir, image_id)):
                raise ValueError(f"Image {image_id} not found")

            with open(self.labels_file, 'r') as f:
                existing_labels = json.load(f)

            if image_id in existing_labels:
                existing_labels[image_id].update(labels)
            else:
                existing_labels[image_id] = labels

            with open(self.labels_file, 'w') as f:
                json.dump(existing_labels, f, indent=2)

            logger.info(f"Labels updated for image: {image_id}")

        except Exception as e:
            logger.error(f"Error updating labels: {str(e)}")
            raise

    def get_labels(self, image_id: str) -> Dict[str, Any]:
        """Get labels for an image.

        Args:
            image_id: Identifier of the image

        Returns:
            Dictionary containing label data if found, None otherwise

        Raises:
            Exception: If label retrieval fails
        """
        try:
            with open(self.labels_file, 'r') as f:
                all_labels = json.load(f)
                if image_id not in all_labels:
                    return None
                labels = all_labels[image_id]
                if not labels or not any(labels.values()):
                    return None
                return labels
        except Exception as e:
            logger.error(f"Error getting labels: {str(e)}")
            raise

    def export_dataset(self, format: str = 'json') -> str:
        """Export dataset in specified format.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Path to exported file

        Raises:
            ValueError: If export format is unsupported
            Exception: If export operation fails
        """
        try:
            export_path = os.path.join(self.dataset_dir, "export")
            os.makedirs(export_path, exist_ok=True)

            with open(self.labels_file, 'r') as f:
                labels = json.load(f)

            stats = self.get_dataset_stats()

            if format.lower() == 'json':
                export_data = {
                    'metadata': {
                        'total_images': stats['total_images'],
                        'labeled_images': stats['labeled_images'],
                        'unlabeled_images': stats['unlabeled_images'],
                        'discarded_images': stats['discarded_images'],
                        'export_date': datetime.now().isoformat(),
                        'format_version': '1.0'
                    },
                    'labels': labels,
                    'distributions': stats['distributions']
                }

                output_file = os.path.join(
                    export_path,
                    'dataset_export.json'
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        export_data,
                        f,
                        indent=2,
                        ensure_ascii=False
                    )

            elif format.lower() == 'csv':
                import csv
                output_file = os.path.join(
                    export_path,
                    'dataset_export.csv'
                )

                fields = [
                    'image_id',
                    'animal_type',
                    'size',
                    'body_condition',
                    'visible_health_issues',
                    'pregnancy_indicators',
                    'image_quality',
                    'context'
                ]

                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()

                    for image_id, label_data in labels.items():
                        row = {'image_id': image_id}
                        row.update(label_data)
                        writer.writerow(row)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Dataset exported successfully to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error exporting dataset: {str(e)}")
            raise

    def get_model_training_data(self, model_type: str) -> dict:
        """Get dataset formatted for model training.

        Args:
            model_type: Type of model for training data formatting

        Returns:
            Dictionary containing training data and metadata
        """
        try:
            training_data = {
                'images': [],
                'labels': [],
                'metadata': {
                    'model_type': model_type,
                    'generated_at': datetime.now().isoformat(),
                    'total_samples': len(self.labels_file)
                }
            }

            for image_id, label_data in self.labels_file.items():
                if image_id not in self.discarded_file.get("discarded_images", {}):
                    image_path = os.path.join(self.raw_dir, image_id)
                    if os.path.exists(image_path):
                        training_data['images'].append(image_path)
                        training_data['labels'].append(label_data)

            return training_data
        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            return {}

    def get_image(self, image_id: str) -> Image.Image:
        """Get an image by its ID.

        Args:
            image_id: Identifier of the image to retrieve

        Returns:
            PIL Image object if found, None otherwise
        """
        try:
            image_path = os.path.join(self.raw_dir, image_id)
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None

            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return img.copy()
            except Exception as e:
                logger.error(f"Error opening image {image_id}: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error getting image: {str(e)}")
            return None
