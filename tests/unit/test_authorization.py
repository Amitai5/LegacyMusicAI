from datetime import UTC, datetime

import pytest

from legacy_music.authorization import (
    AuthorizationError,
    require_capability,
    require_source_asset,
)
from legacy_music.domain.rights import (
    ApprovalRecord,
    RightsManifest,
    RightsPermissions,
    RightsStatus,
    SourceAssetAuthorization,
)

ApprovedAt = datetime(2026, 1, 1, tzinfo=UTC)


def approved_rights() -> RightsManifest:
    return RightsManifest(
        artist_id="test-artist",
        status=RightsStatus.APPROVED,
        effective_at=ApprovedAt,
        permissions=RightsPermissions(training=True, music_style=True),
        approvals=(
            ApprovalRecord(
                holder="Test fixture owner",
                scope="Synthetic test fixture only",
                evidence_reference="test://approval",
                approved_at=ApprovedAt,
            ),
        ),
        source_assets=(
            SourceAssetAuthorization(
                sha256="a" * 64,
                evidence_reference="test://source",
            ),
        ),
    )


def test_require_capability_with_evidenced_active_grant_allows_action() -> None:
    require_capability(
        approved_rights(),
        "test-artist",
        "training",
        datetime(2026, 8, 24, tzinfo=UTC),
    )


def test_require_capability_without_approval_evidence_fails_closed() -> None:
    rights = approved_rights().model_copy(update={"approvals": ()})

    with pytest.raises(AuthorizationError, match="no verified approval"):
        require_capability(rights, "test-artist", "training", ApprovedAt)


def test_require_capability_for_other_artist_fails_closed() -> None:
    with pytest.raises(AuthorizationError, match="different artist"):
        require_capability(approved_rights(), "other-artist", "training", ApprovedAt)


def test_require_source_asset_requires_exact_hash() -> None:
    with pytest.raises(AuthorizationError, match="not listed"):
        require_source_asset(approved_rights(), "b" * 64)
