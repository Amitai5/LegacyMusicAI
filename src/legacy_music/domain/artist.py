"""Artist-profile domain models."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legacy_music.paths import validate_artist_id


class ArtistModel(BaseModel):
    """Base model for immutable artist configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtistCapabilities(ArtistModel):
    """Features authorized and configured for an artist profile."""

    music_style: bool = True
    singing_voice: bool = False


class RightsProfile(ArtistModel):
    """Location of the artist's machine-readable authorization manifest."""

    manifest: Path = Path("rights.yaml")


class MusicProfile(ArtistModel):
    """Artist-specific music-engine configuration."""

    engine: str = "ace-step"
    adapter: str = "selected"


class VoiceProfile(ArtistModel):
    """Artist-specific voice-engine configuration."""

    engine: str = "soulx"
    default_reference: str = "neutral"


class GenerationDefaults(ArtistModel):
    """Default generation values for one artist."""

    default_duration_seconds: int = Field(default=180, ge=10, le=600)
    default_variants: int = Field(default=1, ge=1, le=16)


class ArtistProfile(ArtistModel):
    """Complete model-independent profile for one musical identity."""

    id: str
    display_name: str = Field(min_length=1, max_length=200)
    capabilities: ArtistCapabilities = ArtistCapabilities()
    rights: RightsProfile = RightsProfile()
    music: MusicProfile = MusicProfile()
    voice: VoiceProfile = VoiceProfile()
    generation: GenerationDefaults = GenerationDefaults()

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Require canonical artist identifiers at the model boundary."""
        return validate_artist_id(value)
