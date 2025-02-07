# EPT Vision 🐾

EPT Vision is an advanced computer vision system that uses artificial intelligence to:
- 🏥 Detect animal diseases and health issues
- 📊 Assess body condition and nutrition status
- 🤰 Identify pregnancy indicators
- 🔍 Analyze image quality and context
- 🏷️ Provide automated pre-labeling

## System Architecture

### Tech Stack
- **Backend**: FastAPI
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: PyTorch with EfficientNet B0
- **Database**: PostgreSQL
- **Storage**: AWS S3
- **Containers**: Docker
- **Documentation**: Swagger/OpenAPI

### Core Components

#### 1. REST API
- FastAPI-based endpoints for image processing
- Automatic documentation with Swagger/OpenAPI
- Efficient file handling and storage
- Comprehensive error handling

#### 2. AI Models
- Multi-task learning architecture with EfficientNet B0 backbone
- Task-specific prediction heads for:
  - Animal Type Classification (93.94% accuracy)
  - Size Estimation (73.60% accuracy)
  - Body Condition Assessment (88.79% accuracy)
  - Health Issues Detection (91.02% accuracy)
  - Pregnancy Indicators (61.27% accuracy)
  - Image Quality Analysis (82.76% accuracy)
  - Context Recognition (94.62% accuracy)
- Advanced image preprocessing
- Confidence scoring for all predictions

#### 3. Image Labeling System
- Intuitive web interface for image labeling
- Keyboard shortcuts for rapid labeling
- Batch upload with drag-and-drop support
- Real-time AI predictions
- Offline support with local storage
- Export functionality (JSON/CSV)
- Progress tracking and statistics

#### 4. Infrastructure
- Scalable cloud deployment
- Automatic load balancing
- Monitoring and logging
- Redis caching layer

## Requirements

- Python 3.9.0+
- Docker
- kubectl
- AWS CLI
- Redis
- Node.js 16+ (for development)

## Installation

```bash
# Clone repository
git clone https://github.com/dassantoss/ept-vision.git
cd ept-vision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Generate secure SECRET_KEY
docker compose run --rm api python scripts/generate_secret_key.py
```

## Database Setup

The project uses PostgreSQL as its database. Follow these steps to set up the database:

1. **Verify Database Connection**:
```bash
# Test connection to the database
docker compose run --rm api python scripts/verify_db_connection.py
```

2. **Run Database Migrations**:
```bash
# Apply all pending migrations
docker compose run --rm api alembic upgrade head
```

3. **Environment Variables**:
Make sure your `.env` file contains the correct database configuration:
```env
POSTGRES_SERVER=your-db-host
POSTGRES_USER=your-user
POSTGRES_PASSWORD=your-password
POSTGRES_DB=your-database
POSTGRES_PORT=5432

# Redis Configuration
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=your-region
S3_BUCKET=your-bucket-name
```

### Security Notes

1. **Secret Key**:
   - Never use the default SECRET_KEY in production
   - Generate a new one using `scripts/generate_secret_key.py`
   - Regenerate periodically for enhanced security
   - Keep your `.env` file secure and never commit it to version control

2. **Database Access**:
   - Ensure proper security group settings for RDS in production
   - Use strong passwords and restrict access to necessary IPs only
   - Regular security audits and updates are recommended

3. **Redis Security**:
   - Enable Redis authentication in production
   - Use SSL/TLS for Redis connections
   - Implement proper network isolation

## Development

```bash
# Run tests
pytest

# Start development server
uvicorn app.main:app --reload

# Start Redis server (if running locally)
redis-server

# Run development environment with Docker
docker compose up -d
```

## Image Labeling Interface

The system includes a web-based interface for image labeling with the following features:

### Key Features
- Keyboard shortcuts for rapid labeling
- Drag-and-drop image upload
- Batch processing capabilities
- AI-powered pre-labeling
- Offline support with auto-sync
- Real-time statistics and progress tracking
- Export functionality (JSON/CSV)

### Keyboard Shortcuts
- **1-3**: Animal Type
- **4-6**: Size
- **7-9**: Body Condition
- **Q-T**: Health Issues
- **Y-I**: Pregnancy Indicators
- **O-P**: Image Quality
- **Z-M**: Context
- **D**: Discard Image
- **Arrow Keys**: Navigate Images
- **Enter**: Save Labels

### Offline Support
The labeling interface works offline with:
- Local storage for unsaved changes
- Automatic synchronization when online
- Progress persistence
- Batch operation queuing

## Project Structure

The project follows this structure:

```
ept-vision/
├── app/
│   ├── api/            # API Endpoints
│   │   ├── v1/
│   │   └── dependencies/
│   ├── core/           # Core Configuration
│   │   ├── config.py
│   │   └── security.py
│   ├── models/         # ML Models
│   │   ├── pet_analysis.py        
│   │   ├── base_model.py          # Base class for all models
│   │   ├── disease_detection/     # Disease detection model
│   │   ├── nutrition_analysis/    # Nutrition analysis model
│   │   ├── pet_detection/        # Pet detection model
│   │   ├── pet_recognition/      # Pet recognition model
│   │   ├── pregnancy_detection/  # Pregnancy detection model
│   │   └── prelabeling/         # Pre-labeling utilities
│   ├── schemas/        # Pydantic Models
│   ├── services/       # Business Logic
│   ├── static/         # Static files (CSS, JS, images)
│   ├── templates/      # HTML templates
│   ├── tools/          # Application-specific tools
│   └── utils/          # Utility Functions
├── aws/                # AWS Configuration and utilities
├── data/               # Data directory
│   ├── datasets/       # ML datasets
│   ├── raw_images/     # Original images
│   └── processed_images/ # Processed images
├── docker/             # Docker Files
│   ├── Dockerfile
│   └── docker-compose.yml
│   ├── deployment.yaml
├── logs/               # Application logs
├── scripts/            # Utility Scripts
│   ├── generate_secret_key.py    # Generate secure SECRET_KEY
│   ├── verify_db_connection.py   # Test database connectivity
│   ├── download_datasets.py      # Download ML datasets
│   └── other scripts...
├── tests/              # Unit & Integration Tests
│   ├── api/
│   ├── models/
│   └── services/
├── alembic/            # Database migrations
├── alembic.ini         # Alembic configuration
├── setup.py           # Package configuration
├── .env.example       # Environment Variables Template
├── .gitignore
├── requirements.txt
└── README.md
```

### Key Directories Explained

- **/app/static**: Contains static assets like CSS, JavaScript, and images used in the web interface
- **/app/templates**: HTML templates for web views
- **/app/tools**: Application-specific tools and utilities for dataset management, image processing, and other helper functions
  - `dataset_manager.py`: Manages dataset operations (upload, labeling, export)
  - Other utility modules for data processing and management
- **/aws**: AWS-related configuration and utilities
- **/data**: Storage for datasets and images
  - **/datasets**: Organized ML training datasets
  - **/raw_images**: Original unprocessed images
  - **/processed_images**: Images after preprocessing
- **/logs**: Application logs and monitoring data

### ML Models Structure

The ML models are organized in the `/app/models` directory:

1. **Main Analysis Model** (`pet_analysis.py`):
   - Unified model for comprehensive pet analysis
   - Multi-task architecture with shared backbone
   - Task-specific prediction heads
   - Confidence scoring system
   - Advanced preprocessing pipeline

2. **Pre-labeling System**:
   - Automated labeling suggestions
   - Confidence assessment
   - Quality validation
   - Real-time predictions

### Model Pipeline Flow

```
Input Image
    │
    ▼
1. Preprocessing
    │  • Image normalization
    │  • Size adjustment
    │  • Quality assessment
    │
    ▼
2. Feature Extraction (EfficientNet B0)
    │  • Shared backbone
    │  • Transfer learning
    │
    ▼
3. Multi-task Analysis
    │  • Animal type
    │  • Size estimation
    │  • Health assessment
    │  • Body condition
    │  • Pregnancy indicators
    │  • Context analysis
    │
    ▼
4. Confidence Scoring
    │  • Per-task confidence
    │  • Overall reliability
    │
    ▼
Final Results
```

## API Documentation

### Pet Analysis API

The Pet Analysis API allows you to integrate our pet analysis capabilities into your own website or application.

#### Main Endpoint
```http
POST https://vision.esperandoporti.com/api/v1/analysis/analyze/pet/{imagen_id}
```

#### Usage Flow

1. **Upload Image**
```http
POST /api/v1/dataset/upload
Content-Type: multipart/form-data

file: [image file]
type: "pets"
```

**Successful Response:**
```json
{
    "filename": "20240206_123456_001234.jpg"
}
```

2. **Analyze Image**
```http
POST /api/v1/analysis/analyze/pet/{filename}
```

**Successful Response:**
```json
{
    "result": {
        "labels": {
            "animal_type": "dog",
            "size": "medium",
            "body_condition": "normal",
            "visible_health_issues": "none",
            "pregnancy_indicators": "none",
            "image_quality": "good",
            "context": "home"
        },
        "confidence": 0.95,
        "health_confidence": 0.88,
        "body_condition_confidence": 0.92,
        "pregnancy_confidence": 0.85,
        "context_confidence": 0.90,
        "quality_confidence": 0.95
    }
}
```

#### JavaScript Integration Example

```javascript
async function analizarMascota(imagenFile) {
    try {
        // 1. Upload image
        const formData = new FormData();
        formData.append('file', imagenFile);
        formData.append('type', 'pets');
        
        const uploadResponse = await fetch('https://vision.esperandoporti.com/api/v1/dataset/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Error uploading image');
        }

        const { filename } = await uploadResponse.json();

        // 2. Analyze image
        const analysisResponse = await fetch(
            `https://vision.esperandoporti.com/api/v1/analysis/analyze/pet/${filename}`,
            { method: 'POST' }
        );

        if (!analysisResponse.ok) {
            throw new Error('Analysis error');
        }

        const result = await analysisResponse.json();
        return result;

    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}
```

#### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Parameter error
- `404 Not Found`: Image not found
- `500 Internal Server Error`: Server error

#### Important Notes

1. **Supported Image Formats**:
   - JPEG/JPG
   - PNG
   - WebP

2. **Maximum File Size**: 10MB

3. **Recommendations for Best Results**:
   - Pet should be centered in the image
   - Good lighting
   - Clear background
   - Minimum recommended resolution: 640x640 pixels

4. **Error Handling**:
   - Always check HTTP status code
   - Implement retries for temporary errors
   - Validate image format and size before upload

For more detailed documentation and examples, visit our [API Documentation Page](https://vision.esperandoporti.com/docs).

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

Esperando por ti - [website](https://vision.esperandoporti.com)

```
