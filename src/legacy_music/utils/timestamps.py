"""UTC timestamp and immutable run-identifier formatting."""

from datetime import UTC, datetime
from uuid import UUID


def require_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime or raise for ambiguous input."""
    if value.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware.")
    return value.astimezone(UTC)


def generation_run_id(created_at: datetime, random_id: UUID) -> str:
    """Create a sortable generation run identifier from supplied deterministic inputs."""
    utc = require_utc(created_at)
    return f"{utc:%Y%m%d-%H%M%S}-{random_id.hex[:8]}"
