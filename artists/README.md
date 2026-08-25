# Artist profiles

Real artist profiles are local, sensitive runtime data and are ignored by Git. Create them through the future `legacy-music artist create` command rather than copying this directory manually.

Each profile will contain its own:

- `artist.yaml` configuration;
- approved rights manifest;
- immutable original recordings;
- normalized audio, stems, lyrics, and analysis;
- training datasets and model adapters;
- voice-reference library;
- presets and evaluation results; and
- audit history.

`_template/` documents the committed configuration contract. It is not an authorized artist and must never contain recordings or real personal data.
