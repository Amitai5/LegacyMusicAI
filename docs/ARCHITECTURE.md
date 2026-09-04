# Architecture

## Design goals

Legacy Music AI is a local orchestration application with replaceable model engines. The architecture prioritizes artist isolation, immutable source material, explicit rights checks, reproducible runs, resumability, and the ability to add an artist without changing application code.

## Boundaries

| Boundary | Owns | Must not own |
| --- | --- | --- |
| CLI commands | Input validation, operator prompts, and result presentation | Model logic or direct filesystem layout decisions |
| Domain | Artist, song, rights, training, voice, and generation contracts | Subprocesses, YAML parsing, or audio IO |
| Manifests | Strict serialization and durable contract versions | Business orchestration |
| Pipelines | Ordered workflows, child-run orchestration, and durable stage transitions | Vendor-specific command syntax |
| Engines | ACE-Step, SoulX-Singer, Seed-VC, and separator adapters | Artist discovery or cross-profile access |
| Audio | Validation, normalization, separation results, alignment, and mixing | Rights decisions |
| Repositories | Artist catalogs, run manifests, atomic persistence, and path confinement | Model inference |

Dependencies point inward: commands and pipelines depend on domain contracts and engine protocols; vendor adapters implement those protocols. Upstream repositories never become the source of artist selection, permissions, or run identity.

## Artist isolation

Every data-dependent operation starts with an explicit canonical `artist_id`. `ProjectPaths.artist_root()` validates the identifier and resolves exactly one direct child beneath `artists/`. Pipelines receive the resolved artist root and may not search sibling profiles.

Shared foundation weights belong in `models/shared/`. Artist-specific recordings, tensors, adapters, references, presets, and evaluations belong only within that artist's local root.

## Repository boundary

Version control contains the application workflow and safe scaffolding, not runtime content. The committed artist surface is limited to `artists/README.md` and the configuration-only `artists/_template/`; all real artist profiles are ignored recursively. The committed model surface is limited to `models/README.md`, which documents the runtime layout; downloaded weights, caches, checkpoints, and trained models remain local. Generated runs and installed upstream checkouts are also excluded from Git.

Path-based ignore rules are reinforced by common trained-model filename patterns so a misplaced checkpoint is not staged accidentally. These rules are a safeguard rather than an authorization control: operators must still review staged content and must never force-add private or restricted artifacts.

## Immutable inputs and runs

Source recordings are copied into `data/raw/originals/<song-id>/` and addressed by SHA-256. Preprocessing writes derived files to separate directories. No pipeline stage receives permission to mutate an original path.

Every generation creates a unique run directory. State transitions are append-only and atomic:

```text
CREATED -> MUSIC_GENERATED -> VOCALS_SEPARATED -> VOICE_CONVERTED
        -> MIXED -> COMPLETE
```

A failure records the failed stage while retaining prior artifacts. Resume and reproduction are represented in the contracts and remain backlog command work.

## Environment isolation

The application package remains lightweight apart from the bundled FFmpeg runtime. ACE-Step,
SoulX-Singer, and optional Seed-VC use pinned, isolated upstream checkouts and environments.
ACE generation uses a loopback API; the default managed process exits before voice conversion
to make the sequential pipeline fit the tested 8 GB GPU. Preprocessing/conversion use
argument-list subprocesses with project-local caches. Seed-VC may download supporting weights
on first use. Outputs and provenance retain model, adapter, seed, input, and artifact hashes.

## Independent music, voice, and mix stages

ACE-Step trains music-style adapters from authorized catalog material or verified
accompaniment stems. Seed-VC trains voice identity from separately screened clean lead-vocal
excerpts. Dataset policy distinguishes permanent source exclusions, music-only exclusions,
and voice-only exclusions/ranges.

`vocals.wav` is preserved alongside a separate `clean_vocals.wav`. Cleanup is controlled by
`training_vocals`; final presence, reverb, and bus balance use `final_vocal_mix` instead.
Changing a mix does not contaminate training data. Quality-first remixing creates a new run
using the same parent music and stems, enabling a controlled comparison of vocal processing.

The typed `VoiceConversionResult` carries audio, reference IDs, phrase decisions, metrics,
and a versioned manifest. The additive voice manifest is currently version 4; legacy
references and supported prior manifests remain loadable. Identity-window metrics are
screening evidence rather than an assertion of perceptual identity or release approval.

## Safety gate

The rights manifest is a domain input, not optional metadata. Ingest, preparation, training, voice conversion, generation, and release checks must each validate the capability relevant to that action. A pending, expired, revoked, missing, mismatched, or unverified manifest fails closed.
