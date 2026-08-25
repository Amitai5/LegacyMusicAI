# Dataset preparation

## Source policy

Only authorized source assets whose SHA-256 hashes appear in the artist rights manifest may enter a dataset. Originals are copied, never moved, and never modified after import.

## Planned stages

1. Discover supported audio recursively.
2. Hash files and detect duplicates before copying.
3. Inspect format, duration, channels, sample rate, and integrity with `ffprobe`.
4. Assign stable song IDs and persist catalog entries atomically.
5. Normalize derived copies without destructive mastering.
6. Separate vocals and accompaniment and estimate F0.
7. Transcribe lyrics and extract assistive musical metadata.
8. Curate clean vocal references manually.
9. Create deterministic training and evaluation splits.
10. Cache reusable ACE-Step preprocessing tensors.

Automatic transcription, tempo, key, instrumentation, and style analysis remain assistive metadata and require review where practical.
