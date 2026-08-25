"""Filesystem repositories for isolated artist and run state."""

from legacy_music.repositories.artist import ArtistRepository
from legacy_music.repositories.catalog import CatalogRepository
from legacy_music.repositories.run import RunRepository
from legacy_music.repositories.training import TrainingRepository
from legacy_music.repositories.voice import VoiceReferenceRepository

__all__ = [
    "ArtistRepository",
    "CatalogRepository",
    "RunRepository",
    "TrainingRepository",
    "VoiceReferenceRepository",
]
