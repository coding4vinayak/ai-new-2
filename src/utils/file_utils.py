"""File utility functions for document processing."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from src.models.document import FileType


# Mapping of file extensions to FileType enum
EXTENSION_MAP = {
    ".pdf": FileType.PDF,
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".tif": FileType.IMAGE,
    ".docx": FileType.DOCX,
    ".txt": FileType.TEXT,
}

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())


def detect_file_type(file_path: str) -> FileType:
    """Detect the file type based on file extension.

    Args:
        file_path: Path to the file.

    Returns:
        FileType enum value.

    Raises:
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension not in EXTENSION_MAP:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return EXTENSION_MAP[extension]


def validate_file(file_path: str) -> bool:
    """Validate that a file exists and has a supported type.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file is valid.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    detect_file_type(file_path)
    return True


def get_temp_path() -> Path:
    """Get a temporary directory path for file processing.

    Returns:
        Path to a temporary directory.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="doc_intel_"))
    return temp_dir


def cleanup_temp_files(temp_path: Optional[Path] = None) -> None:
    """Clean up temporary files and directories.

    Args:
        temp_path: Specific temp path to clean up. If None, cleans all doc_intel_ temp dirs.
    """
    if temp_path and temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


async def save_upload(upload_file: "Any", destination: Path) -> Path:
    """Save an uploaded file to the destination path.

    Args:
        upload_file: FastAPI UploadFile object.
        destination: Destination path for the file.

    Returns:
        Path where the file was saved.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        content = await upload_file.read()
        f.write(content)
    return destination
