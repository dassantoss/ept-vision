#!/usr/bin/env python3
"""
Train a multi-task model for pet detection and analysis.
This script will:
1. Load the labeled dataset from labels.json
2. Prepare data for multi-task learning
3. Train the model with the following tasks:
   - Animal type detection (dog/cat)
   - Size estimation
   - Body condition assessment
   - Health issues detection
   - Pregnancy detection
   - Image quality assessment
   - Context classification
4. Log metrics with Weights & Biases
5. Export the model
"""

import os
import json
from pathlib import Path
import yaml
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
from sklearn.model_selection import train_test_split
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
LABELS_FILE = DATA_DIR / "datasets/labels.json"
DISCARDED_FILE = DATA_DIR / "datasets/discarded_images.json"
RAW_IMAGES_DIR = DATA_DIR / "datasets/raw_images"
OUTPUT_DIR = PROJECT_DIR / "models/pet_analysis"

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Label mappings
LABEL_MAPS = {
    'animal_type': {'dog': 0, 'cat': 1},
    'size': {'small': 0, 'medium': 1, 'large': 2},
    'body_condition': {'underweight': 0, 'normal': 1, 'overweight': 2},
    'visible_health_issues': {'none': 0, 'wounds': 1, 'skin_issues': 2, 'other': 3},
    'pregnancy_indicators': {'none': 0, 'possible': 1, 'visible': 2},
    'image_quality': {'poor': 0, 'medium': 1, 'good': 2},
    'context': {'home': 0, 'street': 1, 'shelter': 2, 'other': 3}
}

class PetDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Provide a default image or skip
            raise
        
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to tensors
        label_tensors = {
            'animal_type': torch.tensor(LABEL_MAPS['animal_type'][self.labels[idx]['animal_type']]),
            'size': torch.tensor(LABEL_MAPS['size'][self.labels[idx]['size']]),
            'body_condition': torch.tensor(LABEL_MAPS['body_condition'][self.labels[idx]['body_condition']]),
            'visible_health_issues': torch.tensor(LABEL_MAPS['visible_health_issues'][self.labels[idx]['visible_health_issues']]),
            'pregnancy_indicators': torch.tensor(LABEL_MAPS['pregnancy_indicators'][self.labels[idx]['pregnancy_indicators']]),
            'image_quality': torch.tensor(LABEL_MAPS['image_quality'][self.labels[idx]['image_quality']]),
            'context': torch.tensor(LABEL_MAPS['context'][self.labels[idx]['context']])
        }
        
        return image, label_tensors

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use EfficientNet as backbone
        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        
        # Remove original classifier
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Task-specific heads
        self.animal_type = nn.Linear(1280, len(LABEL_MAPS['animal_type']))
        self.size = nn.Linear(1280, len(LABEL_MAPS['size']))
        self.body_condition = nn.Linear(1280, len(LABEL_MAPS['body_condition']))
        self.health_issues = nn.Linear(1280, len(LABEL_MAPS['visible_health_issues']))
        self.pregnancy = nn.Linear(1280, len(LABEL_MAPS['pregnancy_indicators']))
        self.image_quality = nn.Linear(1280, len(LABEL_MAPS['image_quality']))
        self.context = nn.Linear(1280, len(LABEL_MAPS['context']))

    def forward(self, x):
        features = self.features(x)
        features = features.mean([2, 3])  # Global average pooling
        
        return {
            'animal_type': self.animal_type(features),
            'size': self.size(features),
            'body_condition': self.body_condition(features),
            'visible_health_issues': self.health_issues(features),
            'pregnancy_indicators': self.pregnancy(features),
            'image_quality': self.image_quality(features),
            'context': self.context(features)
        }

def prepare_data():
    """Load and prepare the dataset."""
    logger.info("Loading dataset...")
    
    try:
        # Verificar que los archivos existen
        if not LABELS_FILE.exists():
            raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}")
        if not DISCARDED_FILE.exists():
            raise FileNotFoundError(f"Discarded images file not found at {DISCARDED_FILE}")
        if not RAW_IMAGES_DIR.exists():
            raise FileNotFoundError(f"Raw images directory not found at {RAW_IMAGES_DIR}")
        
        # Load labels and discarded images
        with open(LABELS_FILE) as f:
            labels = json.load(f)
        
        with open(DISCARDED_FILE) as f:
            discarded = json.load(f)['discarded_images']
        
        # Prepare image paths and labels
        image_paths = []
        image_labels = []
        
        for img_name, label_dict in labels.items():
            if img_name not in discarded:
                img_path = RAW_IMAGES_DIR / img_name
                if img_path.exists():
                    image_paths.append(img_path)
                    image_labels.append(label_dict)
                else:
                    logger.warning(f"Image not found: {img_path}")
        
        if not image_paths:
            raise ValueError("No valid images found for training")
        
        logger.info(f"Found {len(image_paths)} valid images for training")
        
        # Split dataset
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, image_labels, test_size=0.2, random_state=42
        )
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = PetDataset(train_paths, train_labels, train_transform)
        val_dataset = PetDataset(val_paths, val_labels, val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        return train_loader, val_loader
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_model():
    """Train the multi-task model."""
    logger.info("Starting model training...")
    
    try:
        # Initialize wandb
        wandb.init(project="ept-vision", name=f"pet-analysis-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Prepare data
        train_loader, val_loader = prepare_data()
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        model = MultiTaskModel().to(device)
        
        # Loss functions for each task
        criterion = {
            'animal_type': nn.CrossEntropyLoss(),
            'size': nn.CrossEntropyLoss(),
            'body_condition': nn.CrossEntropyLoss(),
            'visible_health_issues': nn.CrossEntropyLoss(),
            'pregnancy_indicators': nn.CrossEntropyLoss(),
            'image_quality': nn.CrossEntropyLoss(),
            'context': nn.CrossEntropyLoss()
        }
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                batch_labels = {k: v.to(device) for k, v in labels.items()}
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Calculate loss for each task
                loss = sum(criterion[task](outputs[task], batch_labels[task]) for task in outputs.keys())
                
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{NUM_EPOCHS} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_losses = []
            task_accuracies = {task: [] for task in outputs.keys()}
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    batch_labels = {k: v.to(device) for k, v in labels.items()}
                    
                    outputs = model(images)
                    loss = sum(criterion[task](outputs[task], batch_labels[task]) for task in outputs.keys())
                    val_losses.append(loss.item())
                    
                    # Calculate accuracy for each task
                    for task in outputs.keys():
                        preds = torch.argmax(outputs[task], dim=1)
                        acc = (preds == batch_labels[task]).float().mean()
                        task_accuracies[task].append(acc.item())
            
            # Log metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            metrics = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            
            for task in task_accuracies.keys():
                task_acc = np.mean(task_accuracies[task])
                metrics[f'{task}_accuracy'] = task_acc
            
            wandb.log(metrics)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Ensure output directory exists
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, OUTPUT_DIR / 'best_model.pth')
        
        # Export to ONNX
        logger.info("Exporting model to ONNX...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        torch.onnx.export(
            model, 
            dummy_input,
            OUTPUT_DIR / 'model.onnx',
            input_names=['input'],
            output_names=['animal_type', 'size', 'body_condition', 'health_issues', 
                         'pregnancy', 'image_quality', 'context'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'animal_type': {0: 'batch_size'},
                         'size': {0: 'batch_size'},
                         'body_condition': {0: 'batch_size'},
                         'health_issues': {0: 'batch_size'},
                         'pregnancy': {0: 'batch_size'},
                         'image_quality': {0: 'batch_size'},
                         'context': {0: 'batch_size'}}
        )
        
        logger.info("Training completed!")
        wandb.finish()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
