#!/usr/bin/env python3
"""EPT Vision package setup.

This module configures the package installation including:
- Package metadata
- Dependencies
- Entry points
- Development requirements
"""

from setuptools import setup, find_packages
from typing import List


def read_requirements(filename: str) -> List[str]:
    """Read requirements from file.

    Args:
        filename: Path to requirements file

    Returns:
        List of requirements
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name="ept-vision",
    version="0.1.0",
    description="Computer vision system for pet health analysis",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="EPT Team",
    author_email="contact@eptteam.com",
    url="https://github.com/eptteam/ept-vision",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        # API
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-multipart==0.0.6",
        "python-jose[cryptography]==3.3.0",
        "passlib[bcrypt]==1.7.4",
        "bcrypt==4.0.1",
        "pydantic==2.4.2",
        "pydantic-settings==2.0.3",
        "email-validator==2.1.0",

        # Database
        "sqlalchemy==2.0.23",
        "psycopg2-binary==2.9.9",
        "alembic==1.12.1",

        # ML & Computer Vision
        "torch==2.0.1",
        "torchvision==0.15.2",
        "pillow==10.1.0",
        "numpy==1.26.2",
        "opencv-python-headless==4.8.1.78",
        "transformers[torch]==4.36.2",
        "ultralytics>=8.0.0",
        "scikit-image==0.21.0",

        # AWS
        "boto3==1.29.3",

        # Development
        "python-dotenv==1.0.0",

        # Redis
        "redis==5.0.1",

        # Templates
        "jinja2==3.1.2",
        "aiofiles==23.2.1",
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.7b0',
            'isort>=5.9.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'pycodestyle>=2.7.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ept-vision=app.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    include_package_data=True,
    zip_safe=False
)
