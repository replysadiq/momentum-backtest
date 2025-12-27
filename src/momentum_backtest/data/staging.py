"""
Data staging module for reproducibility.

Copies external data files into the project's ./data folder,
ensuring runs are portable, auditable, and repeatable.
"""

import shutil
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StagingError(Exception):
    """Raised when data staging fails."""
    pass


def stage_parquet_file(
    source_path: Path,
    project_data_dir: Path,
) -> Path:
    """
    Copy a parquet file into the project's data directory.

    Args:
        source_path: Path to the source parquet file
        project_data_dir: Project's data directory (e.g., ./data)

    Returns:
        Path to the staged (copied) file

    Raises:
        StagingError: If copy fails or source doesn't exist
    """
    if not source_path.exists():
        raise StagingError(f"Source parquet file not found: {source_path}")

    if not source_path.is_file():
        raise StagingError(f"Source path is not a file: {source_path}")

    # Create data directory if needed
    project_data_dir.mkdir(parents=True, exist_ok=True)

    # Destination path preserves original filename
    dest_path = project_data_dir / source_path.name

    # Get source file info
    source_stat = source_path.stat()
    source_size_mb = source_stat.st_size / (1024 * 1024)
    source_mtime = datetime.fromtimestamp(source_stat.st_mtime)

    logger.info(f"Staging parquet file:")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Destination: {dest_path}")
    logger.info(f"  Size: {source_size_mb:.2f} MB")
    logger.info(f"  Last modified: {source_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if already staged with same content
    if dest_path.exists():
        dest_stat = dest_path.stat()
        if (dest_stat.st_size == source_stat.st_size and
            dest_stat.st_mtime >= source_stat.st_mtime):
            logger.info(f"  File already staged (same size and timestamp)")
            return dest_path
        else:
            logger.info(f"  Updating staged file (source is newer or different size)")

    # Copy the file
    try:
        shutil.copy2(source_path, dest_path)
        logger.info(f"  Successfully staged parquet file")
    except Exception as e:
        raise StagingError(f"Failed to copy parquet file: {e}")

    # Verify copy succeeded
    if not dest_path.exists():
        raise StagingError(f"Staged file not found after copy: {dest_path}")

    dest_size = dest_path.stat().st_size
    if dest_size != source_stat.st_size:
        raise StagingError(
            f"Staged file size mismatch: expected {source_stat.st_size}, got {dest_size}"
        )

    return dest_path


def stage_ticker_csv(
    source_path: Path,
    project_data_dir: Path,
) -> Path:
    """
    Copy a ticker CSV file into the project's data directory.

    Args:
        source_path: Path to the source CSV file
        project_data_dir: Project's data directory (e.g., ./data)

    Returns:
        Path to the staged (copied) file

    Raises:
        StagingError: If copy fails or source doesn't exist
    """
    if not source_path.exists():
        raise StagingError(f"Source ticker CSV not found: {source_path}")

    if not source_path.is_file():
        raise StagingError(f"Source path is not a file: {source_path}")

    # Create data directory if needed
    project_data_dir.mkdir(parents=True, exist_ok=True)

    # Destination path preserves original filename
    dest_path = project_data_dir / source_path.name

    # Get source file info
    source_stat = source_path.stat()
    source_size_kb = source_stat.st_size / 1024

    logger.info(f"Staging ticker CSV:")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Destination: {dest_path}")
    logger.info(f"  Size: {source_size_kb:.2f} KB")

    # Check if already staged with same content
    if dest_path.exists():
        dest_stat = dest_path.stat()
        if dest_stat.st_size == source_stat.st_size:
            logger.info(f"  File already staged (same size)")
            return dest_path

    # Copy the file
    try:
        shutil.copy2(source_path, dest_path)
        logger.info(f"  Successfully staged ticker CSV")
    except Exception as e:
        raise StagingError(f"Failed to copy ticker CSV: {e}")

    return dest_path
