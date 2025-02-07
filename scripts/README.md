# Model Training Scripts

This directory contains scripts for training the EPT Vision models.

## Training the Pet Analysis Model

The `train_pet_analysis.py` script trains a multi-task model for:
- Animal type detection (dog/cat)
- Size estimation
- Body condition assessment
- Health issues detection
- Pregnancy detection
- Image quality assessment
- Context classification

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Weights & Biases account for logging (optional)

### Dataset Structure

```
data/
├── datasets/
│   ├── labels.json
│   ├── discarded_images.json
│   └── raw_images/
└── models/
    └── pet_analysis/
```

### Usage

1. Prepare your dataset following the structure above
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python scripts/train_pet_analysis.py
   ```

### Model Architecture

The model uses EfficientNet B0 as backbone with custom heads for each task. See the script for detailed architecture.

### Notes

- Training metrics are logged to Weights & Biases if configured
- Best model is saved based on validation loss
- Model is exported in both PyTorch and ONNX formats 