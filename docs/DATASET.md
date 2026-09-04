# Dataset preparation

## Source policy

Only authorized source assets whose SHA-256 hashes appear in the artist rights manifest may enter a dataset. Originals are copied, never moved, and never modified after import.

The versioned artist dataset policy supports three independent scopes: permanent source exclusions, music-style training exclusions, and lead-voice exclusions. A music-only exclusion removes a song from newly prepared ACE-Step datasets without deleting its authorized source/stems or making its clean lead vocal unavailable to the separate voice-training pipeline. Existing policy files remain valid when `music_songs` is absent.

The policy lives in the ignored artist file `data/dataset-policy.yaml`. It is separate from
`rights.yaml`: authorization is necessary, but does not make a source suitable for training.

| Policy field | Effect |
| --- | --- |
| `source_exclusions` | Reject a recording by SHA-256 or matching filename during ingestion and downstream eligibility checks |
| `music_songs` | Exclude a catalog song from music-style training without excluding its clean singing voice |
| `voice_songs` | Exclude a whole song or reviewed time intervals from lead-vocal training and bank candidates |

Use the catalog's actual song IDs. Keep permanent exclusions even after removing a source
file so regeneration cannot silently reintroduce it. A policy change does not erase audio
or unlearn an already-trained checkpoint: curate invalid artifacts explicitly, prepare a new
dataset, and retrain/select a replacement when its training content changes.

Tempo is not an automatic quality exclusion. When the artist policy leaves `music_songs` empty, every otherwise approved accompaniment—including fast and upbeat material—is eligible for newly prepared music-style datasets. Permanent source-quality exclusions and lead-voice exclusions continue to apply independently.

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
10. Separate source-linked accompaniment, vocals, and RMVPE F0 using `voice prepare-stems`.
11. Preserve `vocals.wav` and derive `clean_vocals.wav` using `voice clean-vocals`.
12. Build the five-role bank and prepare separate clean-only Seed-VC excerpts, with
    policy/quality filtering and held-out validation excerpts where enough candidates exist.

Lead-vocal selection screens voiced density, continuity, relative level, level variation,
spectral content, and center dominance. These are heuristics, not a semantic lead-versus-ad-lib
classifier. Listen to retained excerpts and record missed backing vocals, distant fragments,
bleed, or effects as exclusions. Reference-bank quality approval is not release approval.

Changing cleanup strength requires `voice clean-vocals <artist-id> --rebuild`; old cleaned
audio is not silently reused under different settings. Rebuild reference/training artifacts
from the new clean-stem hashes before training. Final-mix reverb/EQ/compression remain separate.

`scripts/analyze_music_tempo.py` provides an optional, read-only accompaniment ranking report
from a prepared dataset. It does not automatically exclude songs or provide authoritative
musical annotations. Automatic transcription/key analysis, richer annotation, and ACE-Step
train/evaluation splitting remain backlog enhancements. The current workflow never invents
missing lyrics or silently labels a vocal song as instrumental.
