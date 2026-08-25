# Architecture

## Design goals

Legacy Music AI is a local orchestration application with replaceable model engines. The architecture prioritizes artist isolation, immutable source material, explicit rights checks, reproducible runs, resumability, and the ability to add an artist without changing application code.

## Boundaries

| Boundary | Owns | Must not own |
| --- | --- | --- |
| CLI commands | Input validation, operator prompts, and result presentation | Model logic or direct filesystem layout decisions |
| Domain | Artist, song, rights, training, voice, and generation contracts | Subprocesses, YAML parsing, or audio IO |
| Manifests | Strict serialization and durable contract versions | Business orchestration |
| Pipelines | Ordered, resumable workflows and stage transitions | Vendor-specific command syntax |
| Engines | ACE-Step, SoulX-Singer, Seed-VC, and separator adapters | Artist discovery or cross-profile access |
| Audio | Validation, normalization, separation results, alignment, and mixing | Rights decisions |
| Repositories | Artist catalogs, run manifests, atomic persistence, and path confinement | Model inference |

Dependencies point inward: commands and pipelines depend on domain contracts and engine protocols; vendor adapters implement those protocols. Upstream repositories never become the source of artist selection, permissions, or run identity.

## Artist isolation

Every data-dependent operation starts with an explicit canonical `artist_id`. `ProjectPaths.artist_root()` validates the identifier and resolves exactly one direct child beneath `artists/`. Pipelines receive the resolved artist root and may not search sibling profiles.

Shared foundation weights belong in `models/shared/`. Artist-specific recordings, tensors, adapters, references, presets, and evaluations belong only within that artist's local root.

## Immutable inputs and runs

Source recordings are copied into `data/raw/originals/<song-id>/` and addressed by SHA-256. Preprocessing writes derived files to separate directories. No pipeline stage receives permission to mutate an original path.

Every generation creates a unique run directory. State transitions are append-only and atomic:

```text
CREATED -> MUSIC_GENERATED -> VOCALS_SEPARATED -> VOICE_CONVERTED
        -> MIXED -> COMPLETE
```

A failure records the failed stage while retaining prior artifacts. Resume and reproduction are represented in the contracts and remain backlog command work.

## Environment isolation

The application package remains lightweight. ACE-Step and SoulX-Singer use pinned, isolated upstream checkouts and environments. ACE generation uses a loopback API; the default managed process exits before SoulX to make the sequential pipeline fit an 8 GB GPU. SoulX preprocessing and conversion use argument-list subprocesses with private project-local caches. Outputs and provenance retain model, adapter, seed, input, and artifact hashes.

## Safety gate

The rights manifest is a domain input, not optional metadata. Ingest, preparation, training, voice conversion, generation, and release checks must each validate the capability relevant to that action. A pending, expired, revoked, missing, mismatched, or unverified manifest fails closed.
