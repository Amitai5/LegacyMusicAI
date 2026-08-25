"""Streaming file hashing utilities."""

from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a lowercase SHA-256 digest without loading the complete file into memory."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
