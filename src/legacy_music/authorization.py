"""Action-specific authorization checks used by protected workflows."""

from __future__ import annotations

from datetime import UTC, datetime

from legacy_music.domain.rights import RightsManifest


class AuthorizationError(PermissionError):
    """Raised when an artist manifest does not authorize an operation."""


def require_capability(
    rights: RightsManifest,
    artist_id: str,
    capability: str,
    at: datetime | None = None,
) -> None:
    """Require an active, evidenced grant for one artist capability."""
    instant = at or datetime.now(UTC)
    if rights.artist_id != artist_id:
        raise AuthorizationError("The rights manifest belongs to a different artist profile.")
    if not rights.approvals:
        raise AuthorizationError("The rights manifest has no verified approval evidence.")
    if not rights.permits(capability, instant):
        raise AuthorizationError(
            f"Artist '{artist_id}' is not currently authorized for '{capability}'."
        )


def require_source_asset(rights: RightsManifest, sha256: str) -> None:
    """Require an exact source recording hash in the authorization manifest."""
    if not any(asset.sha256 == sha256 for asset in rights.source_assets):
        raise AuthorizationError(
            f"Source asset {sha256} is not listed in the artist rights manifest."
        )
