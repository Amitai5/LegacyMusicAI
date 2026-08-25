"""Artist-specific training request models."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legacy_music.paths import validate_artist_id


class MusicTrainingRequest(BaseModel):
    """Validated request for one isolated artist music-adapter run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artist_id: str
    engine: str = "ace-step"
    rank: int = Field(default=32, ge=1, le=256)
    epochs: int = Field(default=100, ge=1, le=10_000)
    save_every: int = Field(default=10, ge=1)
    test_run: bool = False

    @field_validator("artist_id")
    @classmethod
    def validate_artist(cls, value: str) -> str:
        """Require an explicit canonical artist identifier."""
        return validate_artist_id(value)
