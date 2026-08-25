# Specification traceability

This matrix maps the attached 77-section implementation specification to committed core functionality and stable backlog task IDs. It is a coverage aid, not a substitute for task acceptance criteria.

| Specification sections | Subject | Implementation or tasks |
| --- | --- | --- |
| 1-2 | Objective and multi-artist profiles | Domain package, artist template, LM-007 through LM-010, LM-035 |
| 3-7 | Two-stage architecture, model responsibilities, optional engine, hardware | Engine protocols, environment docs, LM-001 through LM-006, LM-020 |
| 8 | Repository layout | Current repository structure |
| 9-13 | Artist configuration, creation, listing, and status | Artist repository and commands; LM-007, LM-008 |
| 14-17 | Immutable originals, song layout, catalog, ingest, normalization | Catalog/ingest pipeline; LM-010 through LM-014 |
| 18-20 | Stem separation, vocal review, reference library | SoulX runtime/reference repository; LM-015, LM-017 |
| 21-24 | Lyrics, musical metadata, ACE dataset, evaluation split | LM-016, LM-019 |
| 25-30 | Shared models, artist adapters, training configuration, quick run, checkpoint evaluation | Pinned installer and ACE training client; LM-003, LM-004, LM-023 through LM-026 |
| 31-32 | Zero-shot voice test and multiple references | SoulX SVC and curated references; LM-020 through LM-022 |
| 33-41 | Generation UX/request/pipeline, typed domains, replaceable interfaces, reference selection | Working CLI/pipeline and engine protocols; LM-020, LM-028 through LM-034 |
| 42 | Primary synchronized MVP strategy | LM-030 |
| 43-44 | Score-controlled and advanced composition modes | LM-038 |
| 45 | Timing-safe mixing | Lossless audio service; LM-030 |
| 46-51 | Immutable runs, provenance, inspection, reproduction, variants | Durable generation runs and inspection; remaining LM-028, LM-031, LM-033, LM-034 work |
| 52-55 | Presets, music/voice evaluation, optional voice fine-tuning | LM-021, LM-022, LM-026, LM-027, LM-033 |
| 56-60 | App/engine configuration, environment isolation, diagnostics | Pinned environments and strict doctor; LM-001 through LM-006 |
| 61-64 | Preparation, state, error handling, structured logging | LM-018, LM-024, LM-028, LM-034 |
| 65-66 | Git strategy and storage | `.gitignore`, runtime READMEs, LM-037 |
| 67-70 | Build order, command surface, full workflow, MVP UX | Complete milestone plan; LM-008, LM-013, LM-018, LM-023, LM-033, LM-034 |
| 71-72 | Future API and multi-artist web interface | LM-039, LM-040 |
| 73 | Mandatory artist isolation | Confined repositories and serialized adapter state; LM-010, LM-035, LM-036 |
| 74 | Future person versus musical-project split | LM-041 |
| 75 | Generic naming | Current domain terminology and review rule |
| 76 | Final technology stack | `pyproject.toml`, config, environment docs, LM-003 through LM-006 |
| 77 | Definition of done | LM-036 plus milestone evidence |

Rights enforcement, release review, memorization checks, and disclosure are stronger requirements carried forward from the repository's authorization-first charter. They are implemented through LM-009, LM-031, and LM-032 and apply across the mapped specification.
