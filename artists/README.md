# Artist profiles

Real artist profiles are local, sensitive runtime data and are ignored recursively by Git. Only this README and the configuration-only `_template/` contract are eligible for version control. Create profiles with `legacy-music artist create` rather than copying this directory manually.

Each profile contains its own runtime data:

- `artist.yaml` configuration;
- approved rights manifest;
- immutable original recordings;
- normalized audio, source-linked stems, private lyrics, and analysis;
- independent source, music-style, and lead-vocal exclusions in `data/dataset-policy.yaml`;
- ACE-Step training datasets and music-style adapters;
- separate `vocals.wav` and `clean_vocals.wav` derivatives;
- five-role voice-reference bank and optional Seed-VC datasets, checkpoints, and selection;
- presets and evaluation results; and
- audit history.

The ignore boundary applies to every other child of `artists/`, regardless of file type. `_template/` is not an authorized artist and must never contain recordings, real authorization evidence, or personal data. Never bypass the boundary with `git add --force`.

Music-style eligibility and lead-vocal eligibility are independent. A fast song may remain
useful for voice training even when excluded from a particular style dataset. Permanent
source exclusions are separate and must remain recorded to prevent re-ingestion. See
[dataset preparation](../docs/DATASET.md) and [voice conversion](../docs/VOICE.md).

Training cleanup never changes the original separated vocal stem. Final-mix EQ, gain,
compression, and restored reverb are configured separately and do not rewrite training clips.
