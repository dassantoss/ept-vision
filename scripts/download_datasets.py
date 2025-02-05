#!/usr/bin/env python3
"""Dataset download script.

This script provides functionality to download and extract pet image datasets:
- Stanford Dogs Dataset
- Oxford-IIIT Pet Dataset

The script includes:
- Robust download handling with retries
- Progress tracking
- Automatic extraction
- Error handling and logging
"""

import os
import requests
import tarfile
import zipfile
from pathlib import Path
import logging
from tqdm import tqdm
import time
from typing import Dict, Any
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset configuration
DATASETS: Dict[str, Dict[str, Any]] = {
    'stanford': {
        'url': 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
        'filename': 'stanford_dogs.tar',
        'extract_dir': 'stanford_dogs',
        'description': 'Stanford Dogs Dataset - 120 dog breeds'
    },
    'oxford': {
        'url': 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
        'filename': 'oxford_pets.tar.gz',
        'extract_dir': 'oxford_pets',
        'description': 'Oxford-IIIT Pet Dataset - 37 pet categories'
    }
}


def create_session_with_retries() -> requests.Session:
    """Create a requests session with retry configuration.

    Returns:
        Configured requests Session object with retry handling
    """
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def download_file(url: str, filename: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress tracking and retry handling.

    Args:
        url: URL to download from
        filename: Path to save the file
        chunk_size: Size of chunks for streaming download

    Returns:
        True if download successful, False otherwise

    Raises:
        Exception: If download fails after all retries
    """
    session = create_session_with_retries()

    for attempt in range(3):
        try:
            response = session.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            filename = str(filename)
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
            logger.error(
                f"Download attempt {attempt + 1} failed: {str(e)}"
            )
            if attempt < 2:
                wait_time = 5 * (attempt + 1)
                logger.info(
                    f"Waiting {wait_time} seconds before retrying..."
                )
                time.sleep(wait_time)
                continue
            raise

    return False


def download_and_extract_dataset(
    dataset_name: str,
    base_dir: Path
) -> None:
    """Download and extract a specific dataset.

    Args:
        dataset_name: Name of the dataset to download
        base_dir: Base directory for dataset storage

    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = DATASETS[dataset_name]
    output_dir = base_dir / dataset['extract_dir']

    if output_dir.exists():
        logger.info(
            f"Dataset directory {output_dir} already exists, skipping..."
        )
        return

    logger.info(
        f"Downloading {dataset['description']} from {dataset['url']}"
    )

    try:
        if download_file(dataset['url'], base_dir / dataset['filename']):
            logger.info("Extracting files...")

            with tarfile.open(base_dir / dataset['filename']) as tar:
                tar.extractall(path=output_dir)

            os.remove(base_dir / dataset['filename'])
            logger.info(
                f"{dataset_name} dataset downloaded and extracted to "
                f"{output_dir}"
            )
    except Exception as e:
        logger.error(f"Error processing {dataset_name} dataset: {str(e)}")
        if output_dir.exists():
            logger.info(f"Cleaning up {output_dir}")
            import shutil
            shutil.rmtree(output_dir)
        raise


def main() -> None:
    """Main function to orchestrate dataset downloads.

    Downloads and extracts all configured datasets to the data directory.
    """
    base_dir = Path('data/datasets/raw_images')
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting dataset downloads to {base_dir}")

    for dataset_name in DATASETS:
        try:
            logger.info(
                f"Processing {DATASETS[dataset_name]['description']}..."
            )
            download_and_extract_dataset(dataset_name, base_dir)
        except Exception as e:
            logger.error(
                f"Failed to process {dataset_name} dataset: {str(e)}"
            )
            continue

    logger.info("Dataset download process completed")


if __name__ == "__main__":
    main()
