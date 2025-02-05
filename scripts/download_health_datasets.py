#!/usr/bin/env python3
"""Health dataset download script.

This script downloads and extracts animal health datasets:
- Cat Diseases Dataset (12 common cat diseases)
- Animal Pathology Dataset (veterinary diagnoses)

The script includes:
- Progress tracking for downloads
- Automatic extraction of ZIP files
- Dataset information file generation
- Error handling and logging
"""

import os
import logging
from pathlib import Path
import requests
import zipfile
import shutil
from typing import Dict, Any
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
HEALTH_DATASETS: Dict[str, Dict[str, Any]] = {
    'cat_diseases': {
        'url': (
            'https://github.com/WZMIAOMIAO/deep-learning-for-image-'
            'processing/raw/master/pytorch_classification/Test11_'
            'efficientnet/cat_12_diseases.zip'
        ),
        'filename': 'cat_diseases.zip',
        'extract_dir': 'cat_diseases',
        'description': 'Dataset of 12 common cat diseases with images'
    },
    'animal_pathology': {
        'url': (
            'https://github.com/TensorFlow/datasets/raw/master/'
            'tensorflow_datasets/images/animal_pathology/'
            'animal_pathology_dataset.zip'
        ),
        'filename': 'animal_pathology.zip',
        'extract_dir': 'animal_pathology',
        'description': 'Animal pathology dataset with veterinary diagnoses'
    }
}


def download_file(url: str, filename: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress tracking.

    Args:
        url: URL to download from
        filename: Path to save the file
        chunk_size: Size of chunks for streaming download

    Returns:
        True if download successful, False otherwise

    Raises:
        requests.exceptions.RequestException: If download fails
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                pbar.update(size)
        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False


def download_github_dataset(dataset_name: str, base_dir: Path) -> bool:
    """Download and extract a dataset from GitHub.

    Args:
        dataset_name: Name of the dataset to download
        base_dir: Base directory for dataset storage

    Returns:
        True if download and extraction successful

    Raises:
        ValueError: If dataset_name is not recognized
        Exception: If download or extraction fails
    """
    if dataset_name not in HEALTH_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = HEALTH_DATASETS[dataset_name]
    output_dir = base_dir / dataset['extract_dir']
    zip_path = base_dir / dataset['filename']

    if output_dir.exists():
        logger.info(
            f"Dataset {dataset_name} already exists in {output_dir}"
        )
        return True

    logger.info(f"Downloading dataset {dataset_name}")
    logger.info(f"Description: {dataset['description']}")

    try:
        if download_file(dataset['url'], zip_path):
            logger.info("Extracting files...")

            output_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            os.remove(zip_path)

            # Create dataset information file
            info_file = output_dir / "dataset_info.txt"
            with open(info_file, "w") as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Description: {dataset['description']}\n")
                f.write(f"Source: {dataset['url']}\n")
                f.write(f"Download Date: {datetime.now().isoformat()}\n")

            logger.info(
                f"Dataset {dataset_name} downloaded and extracted to "
                f"{output_dir}"
            )
            return True

    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if zip_path.exists():
            os.remove(zip_path)
        raise


def main() -> None:
    """Main function to orchestrate health dataset downloads.

    Downloads and extracts all configured health datasets to the data directory.
    Creates necessary directories and handles cleanup on failure.
    """
    base_dir = Path('data/datasets/raw_images')
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting health dataset downloads from GitHub")
    logger.info("Available datasets:")
    for name, info in HEALTH_DATASETS.items():
        logger.info(f"- {name}: {info['description']}")

    for dataset_name in HEALTH_DATASETS:
        try:
            logger.info(f"\nProcessing dataset: {dataset_name}")
            download_github_dataset(dataset_name, base_dir)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {str(e)}")
            continue

    logger.info("Health dataset download process completed")


if __name__ == "__main__":
    main()
