#!/usr/bin/env python3
"""
Test script for the advanced pet analysis model.
"""

import logging
import os
from pathlib import Path
from PIL import Image
from app.models.pet_analysis import model as pet_analysis_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_inference(image_path: str):
    """Test inference with the advanced model on a single image."""
    try:
        # Load image
        logger.info(f"Loading image from: {image_path}")
        image = Image.open(image_path)
        
        # Run inference with advanced model
        logger.info("Running advanced model inference...")
        results = pet_analysis_model.analyze_image(image)
        
        # Print results in a formatted way
        logger.info("\nAdvanced Model Results:")
        logger.info("-" * 50)
        
        if "error" in results:
            logger.error(f"Error during inference: {results['error']}")
            return
            
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"\n{key.upper()}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    # Use the image path provided as an argument or ask for it
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        logger.info("Please provide the absolute path to a test image:")
        test_image = input("> ")
    
    test_inference(test_image) 