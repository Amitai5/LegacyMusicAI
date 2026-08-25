"""Immutable artist dataset and selected ACE-Step adapter persistence."""

from __future__ import annotations

import hashlib
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.domain.catalog import ArtistCatalog
from legacy_music.domain.rights import RightsManifest
from legacy_music.domain.training import (
    SelectedMusicAdapter,
    TrainingDatasetManifest,
    TrainingDatasetSong,
)
from legacy_music.paths import validate_artist_id
from legacy_music.persistence import PersistenceError, dump_json_atomic, file_lock, load_json
from legacy_music.repositories.artist import confined_path
from legacy_music.utils.hashing import sha256_file


class TrainingRepositoryError(RuntimeError):
    """Raised when training state cannot be prepared or resolved safely."""


class TrainingRepository:
    """Keep datasets, training output, and adapter selection inside one artist root."""

    def __init__(self, artist_root: Path, artist_id: str) -> None:
        self.artist_root = artist_root.resolve()
        self.artist_id = validate_artist_id(artist_id)
        self.datasets_root = self.artist_root / "datasets/ace-step"
        self.music_root = self.artist_root / "models/music"

    def prepare_dataset(
        self,
        catalog: ArtistCatalog,
        rights: RightsManifest,
        custom_tag: str,
        caption: str,
        is_instrumental: bool,
        lyrics_dir: Path | None = None,
    ) -> TrainingDatasetManifest:
        """Copy verified normalized songs and explicit metadata into an immutable dataset."""
        require_capability(rights, self.artist_id, "training")
        if catalog.artist_id != self.artist_id or not catalog.songs:
            raise TrainingRepositoryError("The artist catalog contains no trainable songs.")
        if not custom_tag.strip() or not caption.strip():
            raise TrainingRepositoryError("Dataset tag and caption cannot be blank.")

        sources: list[tuple[object, Path, Path | None]] = []
        for song in catalog.songs:
            require_source_asset(rights, song.source_sha256)
            if song.normalized_path is None or song.normalized_sha256 is None:
                raise TrainingRepositoryError(f"Song '{song.id}' has no normalized artifact.")
            normalized = confined_path(self.artist_root, song.normalized_path)
            if not normalized.is_file() or sha256_file(normalized) != song.normalized_sha256:
                raise TrainingRepositoryError(f"Song '{song.id}' failed its integrity check.")
            lyrics = None
            if not is_instrumental:
                if lyrics_dir is None:
                    raise TrainingRepositoryError(
                        "Vocal datasets require --lyrics-dir with one UTF-8 file per song."
                    )
                candidates = (
                    lyrics_dir / f"{Path(song.source_filename).stem}.txt",
                    lyrics_dir / f"{song.id}.txt",
                )
                lyrics = next((path for path in candidates if path.is_file()), None)
                if lyrics is None:
                    raise TrainingRepositoryError(f"Lyrics are missing for song '{song.id}'.")
                if not lyrics.read_text(encoding="utf-8").strip():
                    raise TrainingRepositoryError(f"Lyrics are blank for song '{song.id}'.")
            sources.append((song, normalized, lyrics))

        now = datetime.now(UTC)
        dataset_id = f"dataset-{now:%Y%m%dt%H%M%Sz}-{uuid4().hex[:8]}"
        root = self.datasets_root / dataset_id
        audio_dir = root / "audio"
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        try:
            with file_lock(self.datasets_root / ".datasets.lock"):
                root.mkdir()
                (root / ".creating").touch()
                audio_dir.mkdir()
                records = []
                for song, normalized, lyrics in sources:
                    audio = audio_dir / f"{song.id}.wav"
                    shutil.copyfile(normalized, audio)
                    if sha256_file(audio) != song.normalized_sha256:
                        raise TrainingRepositoryError(
                            f"Dataset copy hash mismatch for '{song.id}'."
                        )
                    caption_file = audio.with_suffix(".caption.txt")
                    caption_file.write_text(
                        f"{custom_tag.strip()}, {caption.strip()}\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    lyrics_file = None
                    if lyrics is not None:
                        lyrics_file = audio.with_suffix(".lyrics.txt")
                        lyrics_file.write_text(
                            lyrics.read_text(encoding="utf-8").strip() + "\n",
                            encoding="utf-8",
                            newline="\n",
                        )
                    records.append(
                        TrainingDatasetSong(
                            song_id=song.id,
                            audio_file=audio.relative_to(self.artist_root),
                            audio_sha256=sha256_file(audio),
                            caption_file=caption_file.relative_to(self.artist_root),
                            lyrics_file=(
                                lyrics_file.relative_to(self.artist_root)
                                if lyrics_file is not None
                                else None
                            ),
                        )
                    )
                manifest = TrainingDatasetManifest(
                    dataset_id=dataset_id,
                    artist_id=self.artist_id,
                    created_at=now,
                    custom_tag=custom_tag.strip(),
                    caption=caption.strip(),
                    is_instrumental=is_instrumental,
                    audio_dir=audio_dir.relative_to(self.artist_root),
                    songs=tuple(records),
                )
                dump_json_atomic(root / "dataset.json", manifest.model_dump(mode="json"))
                for record in records:
                    confined_path(self.artist_root, record.audio_file).chmod(stat.S_IREAD)
                (root / ".creating").unlink()
        except Exception as error:
            _remove_incomplete(root)
            if isinstance(error, TrainingRepositoryError):
                raise
            raise TrainingRepositoryError(
                f"Unable to prepare dataset '{dataset_id}': {error}"
            ) from error
        return manifest

    def load_dataset(self, dataset_id: str) -> TrainingDatasetManifest:
        """Load one complete artist-owned dataset manifest."""
        root = self._dataset_root(dataset_id)
        if (root / ".creating").exists():
            raise TrainingRepositoryError(f"Dataset '{dataset_id}' is incomplete.")
        try:
            manifest = TrainingDatasetManifest.model_validate(load_json(root / "dataset.json"))
        except (PersistenceError, ValidationError) as error:
            raise TrainingRepositoryError(f"Invalid dataset '{dataset_id}': {error}") from error
        if manifest.artist_id != self.artist_id or manifest.dataset_id != dataset_id:
            raise TrainingRepositoryError("Dataset identity does not match its directory.")
        return manifest

    def training_root(self, training_id: str) -> Path:
        """Resolve a direct artist-owned training output directory."""
        if not training_id.startswith("training-") or any(char in training_id for char in "/\\"):
            raise TrainingRepositoryError(f"Invalid training ID: {training_id}")
        root = (self.music_root / "training" / training_id).resolve()
        if root.parent != (self.music_root / "training").resolve():
            raise TrainingRepositoryError("Training path escaped the artist profile.")
        return root

    def new_training_id(self) -> str:
        """Create a sortable collision-resistant training identifier."""
        now = datetime.now(UTC)
        return f"training-{now:%Y%m%dt%H%M%Sz}-{uuid4().hex[:8]}"

    def select_adapter(
        self,
        adapter_id: str,
        adapter_path: Path,
        source_training_id: str | None = None,
    ) -> SelectedMusicAdapter:
        """Select a reviewed artist-owned adapter and persist its directory digest."""
        resolved_music_root = self.music_root.resolve()
        exported_root = adapter_path.resolve()
        if resolved_music_root not in exported_root.parents or not exported_root.is_dir():
            raise TrainingRepositoryError("Selected adapter must be inside this artist's models.")
        resolved = _resolve_loadable_adapter(exported_root)
        selected = SelectedMusicAdapter(
            adapter_id=adapter_id,
            artist_id=self.artist_id,
            selected_at=datetime.now(UTC),
            path=resolved.relative_to(self.artist_root),
            sha256=_directory_sha256(resolved),
            source_training_id=source_training_id,
        )
        dump_json_atomic(
            self.music_root / "selected.json",
            selected.model_dump(mode="json"),
        )
        return selected

    def resolve_selected_adapter(self) -> tuple[SelectedMusicAdapter, Path]:
        """Load and integrity-check the currently selected artist adapter."""
        try:
            selected = SelectedMusicAdapter.model_validate(
                load_json(self.music_root / "selected.json")
            )
        except (PersistenceError, ValidationError) as error:
            raise TrainingRepositoryError(f"No valid selected adapter: {error}") from error
        if selected.artist_id != self.artist_id:
            raise TrainingRepositoryError("Selected adapter belongs to another artist.")
        path = confined_path(self.artist_root, selected.path)
        if not path.is_dir() or _directory_sha256(path) != selected.sha256:
            raise TrainingRepositoryError("Selected adapter failed its integrity check.")
        return selected, path

    def _dataset_root(self, dataset_id: str) -> Path:
        if not dataset_id.startswith("dataset-") or any(char in dataset_id for char in "/\\"):
            raise TrainingRepositoryError(f"Invalid dataset ID: {dataset_id}")
        root = (self.datasets_root / dataset_id).resolve()
        if root.parent != self.datasets_root.resolve():
            raise TrainingRepositoryError("Dataset path escaped the artist profile.")
        return root


def _adapter_weight_files(root: Path) -> tuple[Path, ...]:
    patterns = ("*.safetensors", "*.pt", "*.pth")
    return tuple(sorted(path for pattern in patterns for path in root.glob(pattern)))


def _resolve_loadable_adapter(exported_root: Path) -> Path:
    candidates = [
        exported_root,
        *(path.parent for path in exported_root.rglob("adapter_config.json")),
    ]
    valid = []
    for candidate in candidates:
        has_peft_config = (candidate / "adapter_config.json").is_file()
        has_lokr = (candidate / "lokr_weights.safetensors").is_file()
        if (has_peft_config or has_lokr) and _adapter_weight_files(candidate):
            if candidate not in valid:
                valid.append(candidate)
    if len(valid) != 1:
        raise TrainingRepositoryError(
            "Selected export must contain exactly one loadable PEFT or LoKr adapter."
        )
    return valid[0]


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise TrainingRepositoryError("Adapter directory is empty.")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _remove_incomplete(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
    shutil.rmtree(root, ignore_errors=True)
