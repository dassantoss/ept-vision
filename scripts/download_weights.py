#!/usr/bin/env python3
"""
Download and setup model weights for EPT Vision.

This script downloads and organizes the pre-trained model weights required for:
1. Initial animal detection using YOLOv8x-seg
2. Advanced pet analysis model (multi-task learning model)

The weights will be placed in the following structure:
/app/models/weights/
    ├── yolo/
    │   └── best.pt       # YOLOv8x-seg weights for initial detection
    └── advanced/
        └── best_model.pth # Multi-task model for detailed analysis

Usage:
    python scripts/download_weights.py

Notes:
    - You need to have the model URLs configured in your environment
    - Default model versions:
        - YOLOv8x-seg: v1.0.0
        - Advanced model: v1.0.0
    - The script will skip downloads if weights already exist
    - Weights are not included in the repository due to size
    - Get the latest weights from: https://github.com/dassantoss/ept-vision/releases
"""

import os
from pathlib import Path
import shutil
import logging
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "app" / "models" / "weights"
YOLO_DIR = MODELS_DIR / "yolo"
ADVANCED_DIR = MODELS_DIR / "advanced"

# Model versions and URLs
# Replace these with your actual release URLs or use environment variables
YOLO_VERSION = "v1.0.0"
ADVANCED_VERSION = "v1.0.0"
YOLO_URL = os.getenv("YOLO_WEIGHTS_URL", "")  # Set this in your environment
ADVANCED_URL = os.getenv("ADVANCED_WEIGHTS_URL", "")  # Set this in your environment


def download_file(url: str, dest_path: Path, description: str) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: Source URL for the file
        dest_path: Destination path to save the file
        description: Description for the progress bar
    
    Raises:
        requests.RequestException: If download fails
        OSError: If file cannot be written
    """
    if not url:
        logger.error(f"No URL provided for {description}")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def setup_directories():
    """Create necessary directories for model weights if they don't exist."""
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    ADVANCED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Created model weight directories")


def download_weights():
    """
    Download model weights from storage.
    
    This function will:
    1. Create necessary directories
    2. Download YOLO weights if not present
    3. Download advanced model weights if not present
    
    Environment variables required:
    - YOLO_WEIGHTS_URL: URL for YOLOv8x-seg weights
    - ADVANCED_WEIGHTS_URL: URL for advanced model weights
    
    Raises:
        Exception: If download fails or environment is not configured
    """
    try:
        # Create directories
        setup_directories()
        
        # Download YOLO weights
        yolo_weights_path = YOLO_DIR / "best.pt"
        if not yolo_weights_path.exists():
            logger.info(f"Downloading YOLOv8x-seg weights (version {YOLO_VERSION})")
            download_file(YOLO_URL, yolo_weights_path, "Downloading YOLO weights")
        else:
            logger.info("YOLO weights already exist, skipping download")
        
        # Download advanced model weights
        advanced_weights_path = ADVANCED_DIR / "best_model.pth"
        if not advanced_weights_path.exists():
            logger.info(f"Downloading advanced model weights (version {ADVANCED_VERSION})")
            download_file(ADVANCED_URL, advanced_weights_path, "Downloading advanced model weights")
        else:
            logger.info("Advanced model weights already exist, skipping download")
        
        logger.info("Successfully set up model weights")
        
    except Exception as e:
        logger.error(f"Error downloading weights: {str(e)}")
        raise

if __name__ == "__main__":
    download_weights()
