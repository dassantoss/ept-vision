#!/usr/bin/env python3
"""
Configuration settings module.

This module defines application-wide settings using Pydantic BaseSettings,
allowing for environment variable overrides and type validation for all
configuration parameters.
"""

from typing import List, Union

from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings management.

    This class handles all configuration settings for the application,
    including API, database, security, and AWS settings. All settings
    can be overridden using environment variables.

    Attributes:
        API_V1_STR: API version 1 URL prefix
        PROJECT_NAME: Name of the project
        ACCESS_TOKEN_EXPIRE_MINUTES: JWT token expiration time in minutes
        DOMAIN: Application domain name
        BACKEND_CORS_ORIGINS: List of allowed CORS origins
        REDIS_HOST: Redis server hostname
        REDIS_PORT: Redis server port
        POSTGRES_SERVER: PostgreSQL server hostname
        POSTGRES_USER: PostgreSQL username
        POSTGRES_PASSWORD: PostgreSQL password
        POSTGRES_DB: PostgreSQL database name
        POSTGRES_PORT: PostgreSQL server port
        SECRET_KEY: Secret key for JWT encoding
        AWS_ACCESS_KEY_ID: AWS access key ID
        AWS_SECRET_ACCESS_KEY: AWS secret access key
        AWS_REGION: AWS region name
        S3_BUCKET: S3 bucket name
        MODEL_PATH: Path to ML model files
        BATCH_SIZE: Batch size for model inference
        IMAGE_SIZE: Input image size for models
    """

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "EPT Vision"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Domain
    DOMAIN: str = "localhost"

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # Redis settings
    REDIS_HOST: str = "redis"  # Service name in docker-compose
    REDIS_PORT: int = 6379

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(
        cls,
        v: Union[str, List[str]]
    ) -> Union[List[str], str]:
        """
        Validate and assemble CORS origins.

        Args:
            v: Input value to validate (string or list of strings)

        Returns:
            Union[List[str], str]: Validated CORS origins

        Raises:
            ValueError: If the input format is invalid
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str

    # Security
    SECRET_KEY: str

    # AWS Configuration
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET: str

    # ML Model Settings
    MODEL_PATH: str
    BATCH_SIZE: int
    IMAGE_SIZE: int

    class Config:
        """Pydantic model configuration."""
        case_sensitive = True
        env_file = ".env"
        extra = "allow"  # Allow extra fields in environment


settings = Settings()
