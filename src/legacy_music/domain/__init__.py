"""Validated domain models shared by commands and pipelines."""

from legacy_music.domain.artist import ArtistProfile
from legacy_music.domain.generation import GenerationRequest, PipelineStage
from legacy_music.domain.rights import RightsManifest

__all__ = ["ArtistProfile", "GenerationRequest", "PipelineStage", "RightsManifest"]
