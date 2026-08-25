"""Authorization and release-gate domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legacy_music.paths import validate_artist_id


class RightsModel(BaseModel):
    """Base model for immutable authorization records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RightsStatus(StrEnum):
    """Lifecycle state of an artist authorization manifest."""

    PENDING = "pending"
    APPROVED = "approved"
    EXPIRED = "expired"
    REVOKED = "revoked"


class RightsPermissions(RightsModel):
    """Separately granted capabilities for an artist profile."""

    training: bool = False
    music_style: bool = False
    singing_voice: bool = False
    commercial_release: bool = False


class ApprovalRecord(RightsModel):
    """Non-secret reference to one verified approval record."""

    holder: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    evidence_reference: str = Field(min_length=1)
    approved_at: datetime

    @field_validator("approved_at")
    @classmethod
    def validate_approved_at(cls, value: datetime) -> datetime:
        """Require an unambiguous approval instant."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Approval timestamps must be timezone-aware.")
        return value.astimezone(UTC)


class SourceAssetAuthorization(RightsModel):
    """Authorization record for one immutable source asset."""

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_reference: str = Field(min_length=1)


class ReleasePolicy(RightsModel):
    """Required controls on generated-output release."""

    require_human_approval: bool = True
    require_ai_assisted_disclosure: bool = True
    allow_automatic_distribution: bool = False


class RightsManifest(RightsModel):
    """Machine-readable authorization gate for one artist profile."""

    artist_id: str
    status: RightsStatus = RightsStatus.PENDING
    effective_at: datetime | None = None
    expires_at: datetime | None = None
    permissions: RightsPermissions = RightsPermissions()
    approvals: tuple[ApprovalRecord, ...] = ()
    source_assets: tuple[SourceAssetAuthorization, ...] = ()
    release: ReleasePolicy = ReleasePolicy()
    notes: str | None = None

    @field_validator("artist_id")
    @classmethod
    def validate_artist(cls, value: str) -> str:
        """Require an explicit canonical artist identifier."""
        return validate_artist_id(value)

    @field_validator("effective_at", "expires_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require unambiguous UTC instants in persisted authorization records."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Rights timestamps must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_window(self) -> RightsManifest:
        """Require an authorization expiration after its effective instant."""
        if (
            self.effective_at is not None
            and self.expires_at is not None
            and self.expires_at <= self.effective_at
        ):
            raise ValueError("Rights expiration must be after the effective instant.")
        return self

    def permits(self, capability: str, at: datetime) -> bool:
        """Return whether a capability is approved and active at a supplied instant."""
        if capability not in RightsPermissions.model_fields:
            raise ValueError(f"Unknown rights capability: {capability}")
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("Authorization checks require a timezone-aware instant.")
        at = at.astimezone(UTC)
        if self.status is not RightsStatus.APPROVED:
            return False
        if self.effective_at is not None and at < self.effective_at:
            return False
        if self.expires_at is not None and at >= self.expires_at:
            return False
        return bool(getattr(self.permissions, capability))
