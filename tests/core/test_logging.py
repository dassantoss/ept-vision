import pytest
import logging
import os
from pathlib import Path
from unittest.mock import patch

from app.core.logging import setup_logger, CustomFormatter, get_logger

@pytest.fixture
def temp_log_file(tmp_path):
    return str(tmp_path / "test.log")

def test_custom_formatter():
    """Test custom formatter color codes and format string."""
    formatter = CustomFormatter()
    record = logging.LogRecord(
        "test_logger", logging.INFO, "test.py", 10,
        "Test message", (), None
    )
    formatted = formatter.format(record)
    assert "test_logger" in formatted
    assert "INFO" in formatted
    assert "Test message" in formatted

def test_logger_setup_console_only():
    """Test logger setup without file output."""
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_logger_setup_with_file(temp_log_file):
    """Test logger setup with file output."""
    logger = setup_logger("test_logger", log_file=temp_log_file)
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[1], logging.handlers.RotatingFileHandler)
    assert logger.handlers[1].baseFilename == temp_log_file

def test_logger_file_creation(temp_log_file):
    """Test that log files are created and written to."""
    logger = setup_logger("test_logger", log_file=temp_log_file)
    test_message = "Test log message"
    logger.info(test_message)
    
    assert os.path.exists(temp_log_file)
    with open(temp_log_file, 'r') as f:
        content = f.read()
        assert test_message in content

def test_logger_rotation(temp_log_file):
    """Test log file rotation."""
    max_size = 1024  # 1KB
    logger = setup_logger(
        "test_logger",
        log_file=temp_log_file,
        max_size=max_size,
        backup_count=2
    )
    
    # Write enough data to trigger rotation
    long_message = "x" * (max_size // 10)
    for _ in range(20):
        logger.info(long_message)
    
    # Check that backup files were created
    assert os.path.exists(temp_log_file)
    assert os.path.exists(f"{temp_log_file}.1")

def test_get_logger():
    """Test get_logger function."""
    logger = get_logger("test_module")
    assert logger.name == "ept_vision.test_module"
    assert isinstance(logger.handlers[0].formatter, CustomFormatter)

def test_logger_levels():
    """Test different logging levels."""
    with patch('sys.stdout') as mock_stdout:
        logger = setup_logger("test_logger")
        
        # Test all levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Verify that messages were logged
        assert mock_stdout.write.call_count > 0 