# Dataset preparation

## Source policy

Only authorized source assets whose SHA-256 hashes appear in the artist rights manifest may enter a dataset. Originals are copied, never moved, and never modified after import.

## Implemented core stages

1. Discover supported audio recursively.
2. Hash files and detect duplicates before copying.
3. Inspect format, duration, channels, sample rate, and integrity with FFprobe or the SoundFile fallback.
4. Assign stable song IDs and persist catalog entries atomically.
5. Normalize derived copies without destructive mastering.
6. Prepare an immutable artist-scoped ACE-Step directory with explicit caption files.
7. Require one reviewed lyric file per vocal song; instrumental datasets are marked explicitly.
8. Ask the official ACE-Step API to scan and reject unlabeled samples.
9. Cache reusable ACE-Step preprocessing tensors beneath the artist training run.
10. Curate clean vocal references separately and calculate their RMVPE F0 contours.

Automatic transcription, tempo/key analysis, deterministic train/evaluation splits, and richer per-song annotation remain backlog enhancements. The current workflow never invents missing lyrics or silently labels a vocal song as instrumental.
