# Legacy Music AI

Legacy Music AI is an authorization-first, local platform for creating original, clearly disclosed music with the approved musical and vocal identity of legacy artists. It is designed around isolated artist profiles, immutable source recordings, project-specific music adapters, pluggable voice engines, and reproducible generation runs.

> [!IMPORTANT]
> This repository is a foundation scaffold. It does not generate music yet. Training or voice conversion must only use recordings and identities covered by explicit permission from the artist or estate and every applicable recording, composition, performance, name, image, likeness, and voice rights holder.

## Current status

The initial project foundation now provides:

- a Python 3.11 package managed with `uv`;
- a Typer and Rich command-line shell;
- validated application and artist configuration models;
- artist-safe path resolution and engine boundary protocols;
- configuration templates for ACE-Step, SoulX-Singer, and Seed-VC;
- isolated runtime directories for artists, models, vendors, and runs;
- baseline unit tests and continuous integration;
- architecture, safety, dataset, training, voice, and generation documentation; and
- a milestone-based implementation backlog in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Intended pipeline

```text
Rights-cleared recordings
          |
          +--------------------+
          |                    |
          v                    v
  ACE-Step dataset       Vocal extraction
          |                    |
          v                    v
 Artist-specific LoRA    Curated references
          |                    |
          +---------+----------+
                    |
            Generation request
                    |
                    v
           ACE-Step draft song
                    |
                    v
            Vocal separation
                    |
                    v
       SoulX-Singer or Seed-VC
                    |
                    v
       Timing-safe mix and export
                    |
                    v
       WAV + provenance + audit log
```

The music and voice stages are deliberately separate. Each artist receives an independent data root, music adapter, voice-reference library, configuration, presets, and evaluation history. Cross-artist access is forbidden unless a future feature explicitly requests and authorizes it.

## Planned model integrations

- [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) for complete-song generation and artist-specific LoRA adapters.
- [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) as the primary zero-shot singing voice conversion engine.
- [Seed-VC](https://github.com/Plachtaa/seed-vc) as an optional alternative or fine-tuned voice engine.

The orchestration application does not install these large model environments as package dependencies. Each engine is isolated under `vendor/` and invoked through a typed adapter so CUDA, PyTorch, and Python requirements can evolve independently.

## Quick start

Prerequisites:

- Python 3.11;
- [`uv`](https://docs.astral.sh/uv/);
- Git; and
- FFmpeg for future audio commands.

Install the orchestration application and development tools:

```bash
uv sync --locked --group dev
```

Inspect the command shell and local prerequisites:

```bash
uv run legacy-music --help
uv run legacy-music doctor
```

Run the baseline checks:

```bash
uv run ruff check .
uv run pytest
```

The model engines are intentionally not bootstrapped yet. Their installation and smoke tests are tracked as separate tasks because they require large downloads, license review, compatible CUDA environments, and explicit operator approval.

## Repository layout

```text
LegacyMusicAI/
|-- artists/                 # Local artist profiles; real profiles are Git-ignored
|-- config/                  # Application and engine configuration templates
|-- docs/                    # Architecture, operation, safety, and implementation plans
|-- environments/           # Isolated model-environment guidance
|-- models/                  # Shared model cache; contents are Git-ignored
|-- runs/                    # Immutable training, evaluation, and generation runs
|-- scripts/                 # Bootstrap and hardware verification utilities
|-- src/legacy_music/        # Python orchestration package
|   |-- commands/            # Typer command implementations
|   |-- domain/              # Validated domain models
|   |-- engines/             # Replaceable model and separation boundaries
|   |-- manifests/           # Persisted rights, catalog, training, and run contracts
|   |-- pipeline/            # Resumable application workflows
|   `-- utils/               # Focused shared utilities
|-- tests/                   # Unit, integration, and fixture contracts
|-- vendor/                  # Isolated upstream checkouts; contents are Git-ignored
|-- pyproject.toml
`-- uv.lock
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component boundaries, [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the dependency-ordered build plan and live GitHub backlog, and [`docs/SPEC_TRACEABILITY.md`](docs/SPEC_TRACEABILITY.md) for coverage of all 77 specification sections.

## Non-negotiable safeguards

- Originals are copied into immutable storage and are never normalized, renamed, edited, or deleted in place.
- Every data-dependent command requires an explicit `artist_id` or an interactive artist selection.
- An artist command may only resolve paths beneath that artist's root.
- Training and generation remain blocked until the artist's machine-readable authorization manifest is approved and unexpired.
- Synthetic vocals require separate, explicit voice authorization.
- Every generation receives a new immutable run directory and records its prompt, lyrics reference, seed, engines, adapter, voice reference, inputs, hashes, and parent run.
- Outputs are labeled AI-assisted and remain distinguishable from historical recordings.
- No command may automatically publish or distribute generated audio.

Read [`docs/RIGHTS_AND_SAFETY.md`](docs/RIGHTS_AND_SAFETY.md) before adding any artist data.

## Contributing

Do not submit copyrighted audio, artist profiles, consent documents, model weights, private lyrics, credentials, generated songs, or personal data. Open an issue before a large implementation change and include its `LM-###` task identifier in the branch and pull request.

Development work targets the `production` branch through reviewed pull requests. Direct pushes should be reserved for repository administration and initial scaffolding.

## License

No open-source license has been granted. See [`LICENSE`](LICENSE). Upstream model code and weights retain their own licenses and usage restrictions; source-code compatibility does not grant music, voice, likeness, training-data, or output rights.
