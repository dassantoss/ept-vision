#!/usr/bin/env python3
"""Image renaming script.

This script standardizes image filenames in the dataset:
- Renames files using timestamp and sequential counter
- Maintains file creation order
- Updates related JSON files (labels, discarded images)
- Creates backups before modifications
- Generates filename mapping for reference

The new filename format is: YYYYMMDD_HHMMSS_NNNNNN.ext where:
- YYYYMMDD_HHMMSS: Original file timestamp
- NNNNNN: 6-digit sequential counter
- ext: Original file extension
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

# Valid image extensions
VALID_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}


def get_creation_time(file_path: str) -> datetime:
    """Get file creation time or modification time as fallback.

    Args:
        file_path: Path to the file

    Returns:
        Datetime object representing file creation/modification time
    """
    try:
        creation_time = os.path.getctime(file_path)
        return datetime.fromtimestamp(creation_time)
    except Exception:
        modification_time = os.path.getmtime(file_path)
        return datetime.fromtimestamp(modification_time)


def generate_new_filename(
    old_filename: str,
    creation_date: datetime,
    counter: int
) -> str:
    """Generate new filename with standardized format.

    Args:
        old_filename: Original filename
        creation_date: File creation timestamp
        counter: Sequential counter value

    Returns:
        New standardized filename
    """
    ext = os.path.splitext(old_filename)[1].lower()
    timestamp = creation_date.strftime("%Y%m%d_%H%M%S")
    suffix = f"{counter:06d}"
    return f"{timestamp}_{suffix}{ext}"


def load_json_file(file_path: str, default_value: Any) -> Any:
    """Load JSON file with fallback to default value.

    Args:
        file_path: Path to JSON file
        default_value: Value to return if file doesn't exist

    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default_value


def save_json_file(file_path: str, data: Any) -> None:
    """Save data to JSON file with pretty printing.

    Args:
        file_path: Path to save JSON file
        data: Data to save
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_sorted_image_files(
    raw_dir: str
) -> List[Tuple[str, datetime]]:
    """Get list of image files sorted by creation time.

    Args:
        raw_dir: Directory containing image files

    Returns:
        List of tuples containing filename and creation time
    """
    files_with_time = []
    for filename in os.listdir(raw_dir):
        if filename.startswith('.'):  # Skip hidden files
            continue
        file_path = os.path.join(raw_dir, filename)
        if (os.path.isfile(file_path) and
                os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS):
            creation_time = get_creation_time(file_path)
            files_with_time.append((filename, creation_time))

    return sorted(files_with_time, key=lambda x: x[1])


def process_image_file(
    old_filename: str,
    creation_time: datetime,
    counter: int,
    raw_dir: str,
    backup_dir: str,
    labels_data: Dict,
    discarded_data: Dict
) -> Tuple[str, bool]:
    """Process a single image file.

    Args:
        old_filename: Original filename
        creation_time: File creation time
        counter: Current counter value
        raw_dir: Directory containing images
        backup_dir: Backup directory path
        labels_data: Labels dictionary
        discarded_data: Discarded images dictionary

    Returns:
        Tuple containing:
        - New filename
        - Success flag
    """
    file_path = os.path.join(raw_dir, old_filename)
    new_filename = generate_new_filename(old_filename, creation_time, counter)
    new_path = os.path.join(raw_dir, new_filename)

    # Backup original file
    shutil.copy2(file_path, os.path.join(backup_dir, old_filename))

    try:
        # Rename file
        os.rename(file_path, new_path)
        print(f"Renamed: {old_filename} -> {new_filename}")

        # Update labels
        if old_filename in labels_data:
            labels_data[new_filename] = labels_data.pop(old_filename)

        # Update discarded images
        if old_filename in discarded_data["discarded_images"]:
            discarded_data["discarded_images"][new_filename] = (
                discarded_data["discarded_images"].pop(old_filename)
            )

        return new_filename, True

    except Exception as e:
        print(f"Error processing {old_filename}: {str(e)}")
        # Restore from backup
        backup_path = os.path.join(backup_dir, old_filename)
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
        return new_filename, False


def main() -> None:
    """Main function to orchestrate image renaming process.

    Handles:
    - File renaming with standardized format
    - JSON file updates
    - Backup creation
    - Counter management
    - Error handling and recovery
    """
    # Setup paths
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "data" / "datasets"
    raw_dir = dataset_dir / "raw_images"
    labels_file = dataset_dir / "labels.json"
    discarded_file = dataset_dir / "discarded_images.json"
    mapping_file = dataset_dir / "filename_mapping.json"
    counter_file = dataset_dir / "image_counter.txt"

    # Create backup directory
    backup_dir = dataset_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)

    # Load counter
    try:
        with open(counter_file, 'r') as f:
            counter = int(f.read().strip())
    except FileNotFoundError:
        counter = 0

    # Load data files
    labels_data = load_json_file(labels_file, {})
    discarded_data = load_json_file(
        discarded_file,
        {"metadata": {}, "discarded_images": {}}
    )

    # Create mapping dictionary
    filename_mapping = {}

    # Backup original files
    for file_path in [labels_file, discarded_file]:
        if file_path.exists():
            shutil.copy2(file_path, backup_dir / file_path.name)

    # Process files
    files_with_time = get_sorted_image_files(raw_dir)
    initial_counter = counter

    for old_filename, creation_time in files_with_time:
        new_filename, success = process_image_file(
            old_filename,
            creation_time,
            counter,
            raw_dir,
            backup_dir,
            labels_data,
            discarded_data
        )
        if success:
            filename_mapping[old_filename] = new_filename
            counter += 1

    # Save updated data
    save_json_file(labels_file, labels_data)
    save_json_file(discarded_file, discarded_data)

    # Save current counter
    with open(counter_file, 'w') as f:
        f.write(str(counter))

    # Save mapping
    mapping_data = {
        "metadata": {
            "migration_date": datetime.now().isoformat(),
            "total_files": len(filename_mapping),
            "initial_counter": initial_counter,
            "final_counter": counter
        },
        "mapping": filename_mapping
    }
    save_json_file(mapping_file, mapping_data)

    # Print summary
    print(f"\nMigration completed:")
    print(f"- Processed {len(filename_mapping)} files")
    print(f"- Counter range: {initial_counter} -> {counter}")
    print(f"- Backup created in: {backup_dir}")
    print(f"- Mapping saved to: {mapping_file}")


if __name__ == "__main__":
    main()
