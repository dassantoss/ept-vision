# EPT Vision 🐾

EPT Vision is an advanced computer vision system that uses artificial intelligence to:
- 🏥 Detect animal diseases
- 📊 Assess malnutrition states
- 🤰 Identify animal pregnancies
- 🔍 Recognize lost pets
- 🏷️ Automated image labeling

## System Architecture

### Tech Stack
- **Backend**: FastAPI
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: PyTorch with EfficientNetV2 and Vision Transformers (ViT)
- **Database**: PostgreSQL
- **Storage**: AWS S3
- **Cache**: Redis
- **Containers**: Docker + Kubernetes
- **CI/CD**: GitHub Actions

### Core Components

#### 1. REST API
- JWT Authentication
- Optimized endpoints for image processing
- Automatic documentation with Swagger/OpenAPI
- Redis caching for model predictions

#### 2. AI Models
- Advanced image preprocessing techniques
- Multi-task architecture for different types of analysis
- Validation system and performance metrics
- Automated pre-labeling system
- Model prediction caching

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
├── k8s/                # Kubernetes Configuration
│   ├── deployment.yaml
│   └── service.yaml
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
├── run_docker.sh      # Docker run script
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

1. **Detection Models** (First stage - Object Detection):
   - **pet_detection/**: Uses YOLOv8 to detect and locate pets in images
     - Identifies cats and dogs in images
     - Provides bounding box coordinates
     - Estimates animal size
     - Analyzes image quality
     - Counts number of animals

2. **Recognition Models**:
   - **pet_recognition/**: ViT for breed identification
     - Breed classification
     - Confidence scoring
     - Similar breed suggestions

3. **Health Analysis Models**:
   - **disease_detection/**: Health issue identification
   - **nutrition_analysis/**: Nutrition assessment
   - **pregnancy_detection/**: Pregnancy detection

4. **Support Models**:
   - **prelabeling/**: Automated labeling
   - **base_model.py**: Base ML functionality

### Model Pipeline Flow

```
Input Image
    │
    ▼
1. Pet Detection (YOLO)
    │  • Detects animals
    │  • Locates them in image
    │  • Estimates size
    │  • Analyzes image quality
    │
    ▼
2. Pet Recognition (ViT)
    │  • Identifies breed
    │  • Provides confidence scores
    │
    ▼
3. Health Analysis
    │  • Checks for diseases
    │  • Assesses nutrition
    │  • Detects pregnancy
    │
    ▼
4. Pre-labeling
    │  • Generates automatic labels
    │  • Confidence assessment
    │  • Quality validation
    │
    ▼
Final Analysis
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

Esperando por ti - [website](https://esperandoporti.com)

```
