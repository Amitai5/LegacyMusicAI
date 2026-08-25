"""Protocols that isolate upstream model implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from legacy_music.domain.generation import MusicGenerationRequest
from legacy_music.domain.voice import VoiceConversionRequest


class StemResult(BaseModel):
    """Minimum output contract for a stem-separation engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vocals: Path
    accompaniment: Path
    f0: Path | None = None


class MusicEngine(Protocol):
    """Boundary implemented by a complete-song generation engine."""

    def generate(self, request: MusicGenerationRequest, output: Path) -> Path:
        """Generate a complete draft song at the requested output path."""
        ...


class VoiceEngine(Protocol):
    """Boundary implemented by a singing voice conversion engine."""

    def convert(self, request: VoiceConversionRequest, output: Path) -> Path:
        """Convert a singing performance using an authorized voice reference."""
        ...


class StemSeparator(Protocol):
    """Boundary implemented by an audio stem-separation engine."""

    def separate(self, audio: Path, output_dir: Path) -> StemResult:
        """Separate a complete mix into the minimum required stems."""
        ...
