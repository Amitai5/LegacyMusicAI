"""Crash-safe persistence helpers for repository-local manifests."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


class PersistenceError(RuntimeError):
    """Raised when a manifest cannot be persisted safely."""


def dump_json_atomic(path: Path, value: Any) -> None:
    """Serialize JSON and atomically replace the target file."""
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _write_text_atomic(path, text)


def dump_yaml_atomic(path: Path, value: Any) -> None:
    """Serialize YAML and atomically replace the target file."""
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    _write_text_atomic(path, text)


def load_json(path: Path) -> Any:
    """Load a JSON document with a repository-specific error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PersistenceError(f"Unable to load JSON '{path}': {error}") from error


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise PersistenceError(f"Unable to persist '{path}': {error}") from error


@contextmanager
def file_lock(path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Hold a simple exclusive lock file for a short repository transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None

    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise PersistenceError(f"Timed out waiting for lock '{path}'.") from None
            time.sleep(0.05)
        except OSError as error:
            raise PersistenceError(f"Unable to acquire lock '{path}': {error}") from error

    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)
