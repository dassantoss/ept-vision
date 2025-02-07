#!/usr/bin/env python3
"""
Demo script to show how to use both the YOLO and advanced pet analysis models.
"""

import argparse
from pathlib import Path
import logging
from PIL import Image
import cv2
import numpy as np
from app.models.pet_analysis import model as pet_analysis_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_detection(image: np.ndarray, detection: dict) -> np.ndarray:
    """Draw bounding box and confidence on image."""
    x1, y1, x2, y2 = map(int, detection['bbox'])
    conf = detection['confidence']
    
    # Draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw label
    label = f"{detection['class']}: {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1-h-5), (x1+w, y1), (0, 255, 0), -1)
    cv2.putText(image, label, (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def process_image(image_path: str, output_dir: str = 'outputs') -> None:
    """
    Process an image and save results.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output images
    """
    try:
        # Load image
        image = Image.open(image_path)
        logger.info(f"Processing image: {image_path}")
        
        # Run inference
        results = pet_analysis_model.analyze_image(image)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        # Draw detections on image
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Add text with results
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        results_text = [
            f"Animal: {results['animal_type']} ({results['confidence']:.2f})",
            f"Size: {results['size']}",
            f"Body: {results['body_condition']} ({results['body_condition_confidence']:.2f})",
            f"Health: {results['health_issues']} ({results['health_confidence']:.2f})",
            f"Context: {results['context']} ({results['context_confidence']:.2f})"
        ]
        
        for text in results_text:
            cv2.putText(img_cv, text, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
            y_offset += 25
        
        # Save annotated image
        output_file = output_path / f"{Path(image_path).stem}_analysis.jpg"
        cv2.imwrite(str(output_file), img_cv)
        logger.info(f"Results saved to: {output_file}")
        
        # Save analysis results as text
        output_file = output_path / f"{Path(image_path).stem}_analysis.txt"
        with open(output_file, 'w') as f:
            f.write("Analysis Results:\n")
            f.write("-" * 50 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        # Print analysis results
        logger.info("\nAnalysis Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Demo pet analysis models")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", default="outputs", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    process_image(args.image_path, args.output)

if __name__ == "__main__":
    main() 