#!/usr/bin/env python3
"""Synthetic dataset generation script.

This script generates synthetic images for animal health conditions using
Stable Diffusion. The generated dataset includes:
- Images for various health conditions
- Associated metadata and prompts
- Condition-specific directories
- Dataset statistics

Generated conditions include:
- Wounds and injuries
- Malnutrition signs
- Obesity indicators
- Pregnancy signs
- Skin conditions
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Health conditions configuration
HEALTH_CONDITIONS: Dict[str, Dict[str, Any]] = {
    'wounds': {
        'prompts': [
            "close up photo of a dog with a small wound on its leg, "
            "veterinary examination",
            "cat with a minor scratch wound, veterinary clinic setting",
            "puppy with a bandaged wound, veterinary care",
            "close up of a pet wound being treated, veterinary medicine"
        ],
        'num_images': 50,
        'description': 'Images of pet wounds and injuries'
    },
    'malnutrition': {
        'prompts': [
            "thin undernourished stray cat, veterinary examination",
            "malnourished rescue dog, veterinary check up",
            "skinny pet showing signs of malnutrition, veterinary care",
            "underweight animal at vet clinic"
        ],
        'num_images': 50,
        'description': 'Images of pets showing malnutrition signs'
    },
    'obesity': {
        'prompts': [
            "overweight cat at veterinary clinic",
            "obese dog during vet examination",
            "pet with visible signs of obesity, veterinary check up",
            "overweight animal at vet clinic"
        ],
        'num_images': 50,
        'description': 'Images of pets with obesity'
    },
    'pregnancy': {
        'prompts': [
            "pregnant cat at veterinary clinic",
            "pregnant dog during ultrasound examination",
            "late-stage pregnant pet at vet clinic",
            "veterinary examination of pregnant animal"
        ],
        'num_images': 50,
        'description': 'Images of pregnant pets'
    },
    'skin_issues': {
        'prompts': [
            "dog with visible skin rash, veterinary examination",
            "cat with patches of missing fur, veterinary clinic",
            "pet with dermatitis at vet clinic",
            "close up of animal skin condition, veterinary care"
        ],
        'num_images': 50,
        'description': 'Images of pet skin conditions'
    }
}


def setup_model() -> StableDiffusionPipeline:
    """Initialize and configure Stable Diffusion model.

    Returns:
        Configured StableDiffusionPipeline instance

    Raises:
        Exception: If model initialization fails
    """
    try:
        model_id = "stabilityai/stable-diffusion-2-1"

        # Initialize model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None  # Disable for medical images
        )

        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)

        logger.info(f"Model initialized successfully on {device}")
        return pipe

    except Exception as e:
        logger.error(f"Error setting up model: {str(e)}")
        raise


def generate_images(
    pipe: StableDiffusionPipeline,
    condition_name: str,
    condition_info: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate images for a specific health condition.

    Args:
        pipe: Stable Diffusion pipeline instance
        condition_name: Name of the health condition
        condition_info: Configuration for the condition
        output_dir: Base output directory

    Raises:
        Exception: If image generation fails
    """
    try:
        # Create condition directory
        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        # Save condition metadata
        info_file = condition_dir / "condition_info.json"
        with open(info_file, "w") as f:
            json.dump({
                "condition": condition_name,
                "description": condition_info["description"],
                "num_images": condition_info["num_images"],
                "prompts": condition_info["prompts"]
            }, f, indent=2)

        # Generate images
        for i in tqdm(
            range(condition_info["num_images"]),
            desc=f"Generating {condition_name} images"
        ):
            # Select prompt
            prompt = condition_info["prompts"][i % len(condition_info["prompts"])]

            # Generate image
            image = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]

            # Save image
            image_path = condition_dir / f"image_{i:04d}.png"
            image.save(image_path)

            # Save metadata
            meta_path = condition_dir / f"image_{i:04d}.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "prompt": prompt,
                    "condition": condition_name,
                    "image_id": i,
                    "parameters": {
                        "inference_steps": 30,
                        "guidance_scale": 7.5
                    }
                }, f, indent=2)

    except Exception as e:
        logger.error(
            f"Error generating images for {condition_name}: {str(e)}"
        )
        raise


def main() -> None:
    """Main function to orchestrate dataset generation.

    Handles:
    - Output directory setup
    - Model initialization
    - Image generation for all conditions
    - Progress tracking and logging
    """
    # Setup output directory
    output_dir = Path('data/datasets/synthetic_health')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting synthetic health dataset generation")
    logger.info("Conditions to generate:")
    for name, info in HEALTH_CONDITIONS.items():
        logger.info(
            f"- {name}: {info['description']} "
            f"({info['num_images']} images)"
        )

    try:
        # Initialize model
        logger.info("Configuring Stable Diffusion model...")
        pipe = setup_model()

        # Generate images for each condition
        for condition_name, condition_info in HEALTH_CONDITIONS.items():
            try:
                logger.info(f"\nGenerating images for {condition_name}...")
                generate_images(pipe, condition_name, condition_info, output_dir)
            except Exception as e:
                logger.error(
                    f"Failed to generate {condition_name} images: {str(e)}"
                )
                continue

        logger.info("\n✅ Dataset generation completed!")
        logger.info(f"Images saved in: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Dataset generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
