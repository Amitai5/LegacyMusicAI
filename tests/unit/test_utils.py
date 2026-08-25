from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from legacy_music.utils.hashing import sha256_file
from legacy_music.utils.timestamps import generation_run_id, require_utc


def test_sha256_file_streams_expected_digest(tmp_path: Path) -> None:
    value = tmp_path / "value.txt"
    value.write_bytes(b"legacy-music")

    result = sha256_file(value, chunk_size=2)

    assert result == "945ab859fb2ddc5b15ab9e6964871f8c82f3fedbe370dbe048fb41a30432d74c"


def test_generation_run_id_uses_utc_and_stable_suffix() -> None:
    created_at = datetime(2026, 8, 24, 14, 47, 5, tzinfo=UTC)
    random_id = UUID("a81c0000-0000-0000-0000-000000000000")

    assert generation_run_id(created_at, random_id) == "20260824-144705-a81c0000"


def test_require_utc_with_naive_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        require_utc(datetime(2026, 8, 24))
