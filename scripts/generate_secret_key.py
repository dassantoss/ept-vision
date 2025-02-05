#!/usr/bin/env python3
"""Secret key generation script.

This script generates and manages the application's secret key:
- Generates a secure random key using secrets module
- Creates or updates the .env file
- Handles key rotation and backup
- Provides logging and error handling

The secret key is used for:
- Session encryption
- Token generation
- Secure cookie handling
"""

import secrets
import os
from pathlib import Path
from typing import Optional, Tuple
from app.core.logging import get_logger

logger = get_logger("generate_secret_key")


def backup_env_file(env_path: Path) -> Optional[Path]:
    """Create a backup of the existing .env file.

    Args:
        env_path: Path to the .env file

    Returns:
        Path to backup file if created, None otherwise
    """
    try:
        if env_path.exists():
            backup_path = env_path.with_suffix('.env.backup')
            env_path.rename(backup_path)
            logger.info(f"Created backup at {backup_path}")
            return backup_path
        return None
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        return None


def read_env_file(env_path: Path) -> Tuple[list, bool]:
    """Read and parse the .env file.

    Args:
        env_path: Path to the .env file

    Returns:
        Tuple containing:
        - List of environment variable lines
        - Boolean indicating if SECRET_KEY was found
    """
    lines = []
    secret_key_found = False

    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('SECRET_KEY='):
                        secret_key_found = True
        except Exception as e:
            logger.error(f"Error reading .env file: {str(e)}")

    return lines, secret_key_found


def write_env_file(
    env_path: Path,
    lines: list,
    secret_key: str,
    secret_key_found: bool
) -> bool:
    """Write the updated environment variables to .env file.

    Args:
        env_path: Path to the .env file
        lines: List of environment variable lines
        secret_key: New secret key to write
        secret_key_found: Whether SECRET_KEY existed before

    Returns:
        True if write successful, False otherwise
    """
    try:
        new_lines = []
        for line in lines:
            if line.startswith('SECRET_KEY='):
                new_lines.append(f"SECRET_KEY={secret_key}\n")
            else:
                new_lines.append(line)

        if not secret_key_found:
            new_lines.append(f"SECRET_KEY={secret_key}\n")

        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        return True

    except Exception as e:
        logger.error(f"Error writing .env file: {str(e)}")
        return False


def generate_secret_key() -> None:
    """Generate a secure secret key and update the .env file.

    Generates a 32-byte (256-bit) secure random key and saves it to
    the .env file. Creates a new file if it doesn't exist or updates
    the existing SECRET_KEY value.

    Creates a backup of the existing .env file before modifications.

    Raises:
        Exception: If key generation or file operations fail
    """
    try:
        # Generate secure secret key
        secret_key = secrets.token_urlsafe(32)

        # Get path to .env file
        env_path = Path(__file__).parent.parent / '.env'

        # Create backup of existing file
        backup_path = backup_env_file(env_path)

        # Read existing environment variables
        lines, secret_key_found = read_env_file(env_path)

        # Write updated environment variables
        if write_env_file(env_path, lines, secret_key, secret_key_found):
            logger.info("✅ SECRET_KEY updated successfully")
            if backup_path and backup_path.exists():
                backup_path.unlink()
                logger.info("Removed backup file after successful update")
        else:
            if backup_path and backup_path.exists():
                backup_path.rename(env_path)
                logger.info("Restored backup file after failed update")
            raise Exception("Failed to update SECRET_KEY")

    except Exception as e:
        logger.error(f"❌ Error generating SECRET_KEY: {str(e)}")
        raise


if __name__ == "__main__":
    generate_secret_key()
