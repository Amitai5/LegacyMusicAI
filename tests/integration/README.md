# Integration tests

Integration tests exercise complete generation and immutable vocal-remix workflows using
synthetic audio, temporary directories, fake model engines, and the real FFmpeg audio service.
They check timing, mastering, inconsistent vocal levels, current authorization, artifact
lineage, and parent immutability without loading GPU models or using artist recordings.

Run from the repository root:

```powershell
uv run pytest tests/integration
```

The locked application dependencies include a bundled FFmpeg fallback. Real model, GPU,
network, and artist-audio evaluations remain opt-in, private, and outside general CI.
Passing synthetic tests does not validate perceptual voice identity or music quality.
