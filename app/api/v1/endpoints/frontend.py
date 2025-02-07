#!/usr/bin/env python3
"""
Frontend endpoints for the web interface.

This module provides endpoints for serving the web interface
and handling file uploads for the pet analysis system.
"""

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import os
import shutil
from datetime import datetime

from app.core.logging import get_logger

logger = get_logger("ept_vision.frontend")
router = APIRouter()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("/app/data/uploads")
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory initialized at {UPLOAD_DIR}")
    logger.info(f"Upload directory permissions: {oct(UPLOAD_DIR.stat().st_mode)}")
    logger.info(f"Upload directory owner: {UPLOAD_DIR.owner()}")
    logger.info(f"Upload directory group: {UPLOAD_DIR.group()}")
except Exception as e:
    logger.error(f"Error creating upload directory: {str(e)}")

@router.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """
    Serve the pet analysis web interface.

    Args:
        request: FastAPI request object

    Returns:
        HTMLResponse: Rendered analyze.html template
    """
    return templates.TemplateResponse(
        "analyze.html",
        {"request": request}
    )

@router.get("/docs", response_class=HTMLResponse)
async def docs_page(request: Request):
    """
    Serve the API documentation page.

    Args:
        request: FastAPI request object

    Returns:
        HTMLResponse: Rendered docs.html template
    """
    return templates.TemplateResponse(
        "docs.html",
        {"request": request}
    )

@router.get("/presentation", response_class=HTMLResponse)
async def presentation_page(request: Request):
    """
    Serve the project presentation page.

    Args:
        request: FastAPI request object

    Returns:
        HTMLResponse: Rendered presentation.html template
    """
    return templates.TemplateResponse(
        "presentation.html",
        {"request": request}
    )

@router.post("/api/v1/dataset/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handle file upload for analysis.

    Args:
        file: Uploaded file

    Returns:
        JSONResponse with upload status and file info
    """
    try:
        # Log upload attempt and directory status
        logger.info(f"Attempting to upload file: {file.filename}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Upload directory exists: {UPLOAD_DIR.exists()}")
        logger.info(f"Upload directory is directory: {UPLOAD_DIR.is_dir()}")
        if UPLOAD_DIR.exists():
            logger.info(f"Upload directory permissions: {oct(UPLOAD_DIR.stat().st_mode)}")
            logger.info(f"Upload directory owner: {UPLOAD_DIR.owner()}")
            logger.info(f"Upload directory group: {UPLOAD_DIR.group()}")
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        logger.info(f"Generated file path: {file_path}")

        # Ensure directory exists with correct permissions
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(UPLOAD_DIR, 0o777)
        
        # Save uploaded file
        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Set permissions for the uploaded file
            os.chmod(file_path, 0o666)
            
            logger.info(f"File saved successfully at {file_path}")
            logger.info(f"File permissions: {oct(file_path.stat().st_mode)}")
            
            return JSONResponse({
                "success": True,
                "filename": filename,
                "path": str(file_path)
            })
        except Exception as write_error:
            logger.error(f"Error writing file: {str(write_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error writing file: {str(write_error)}"
            )

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )
