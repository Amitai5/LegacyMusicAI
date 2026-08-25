# Legacy Music AI

Create original, clearly disclosed music with the authorized sonic identity of legacy artists—including artists who have passed away—using a small set of rights-cleared reference recordings.

> [!IMPORTANT]
> Legacy Music AI is an early-stage concept, not a working music generator yet. The project is intended for collaborations approved by the artist, their estate, and the relevant recording and publishing rights holders. Public availability of a song does **not** grant permission to train on it or imitate its performer.

## Vision

Many artists leave behind a distinctive musical language and a community that wants to celebrate it. Legacy Music AI aims to help authorized estates, archives, producers, and artists create respectful new works from a limited catalog while keeping people in control of every creative and release decision.

The goal is not to pass generated music off as a newly discovered or authentic performance. Every output should be original, reviewed by a human, traceable to an approved project, and labeled as AI-assisted.

## What the project will do

- Accept a small collection of high-quality, rights-cleared reference songs.
- Record consent, ownership, allowed uses, territories, and expiration dates before training.
- Learn project-specific musical characteristics through a constrained adapter rather than a general-purpose identity clone.
- Generate editable musical ideas such as arrangements, harmonies, instrumentation, and—with separate authorization—synthetic vocals.
- Keep a human producer or estate representative in the approval loop.
- Attach disclosure, provenance, and an audible or machine-readable watermark to exports.
- Preserve an audit trail connecting each model, prompt, source asset, and rendered track to its permissions.

## Authorization-first workflow

1. **Clear the rights** — verify permission from the artist or estate plus the applicable master-recording, composition, name, image, likeness, and voice rights holders.
2. **Register the project** — document allowed uses and explicitly prohibited uses in a machine-readable rights manifest.
3. **Prepare the references** — validate audio quality, split approved material into training and evaluation sets, and prevent unapproved files from entering the pipeline.
4. **Train a scoped adapter** — fine-tune the smallest practical project-specific component and restrict it to the approved identity and purpose.
5. **Generate and review** — create candidates, run similarity and memorization checks, and require human approval.
6. **Export responsibly** — label the work as AI-assisted, embed provenance, retain credits, and honor revocation or expiration rules.

## Planned architecture

| Area | Responsibility |
| --- | --- |
| Rights registry | Consent records, ownership, permitted uses, expiration, and revocation |
| Audio ingestion | Format validation, quality checks, segmentation, and dataset versioning |
| Training pipeline | Reproducible preprocessing, project-scoped adapters, and evaluation |
| Generation service | Controlled inference for music and separately authorized vocals |
| Safety evaluation | Memorization, source similarity, identity misuse, and disclosure checks |
| Review workspace | Human selection, editing, credits, approvals, and release history |
| Provenance layer | Watermarks, manifests, model lineage, and export metadata |

The implementation will favor replaceable model providers so that rights enforcement, auditability, and review rules do not depend on one model vendor.

## Proposed repository layout

```text
LegacyMusicAI/
├── configs/               # Training and inference configuration
├── docs/                  # Architecture, rights, and operating policies
├── src/legacy_music_ai/   # Application and pipeline code
├── tests/                 # Unit, integration, and safety evaluation tests
├── .gitignore
└── README.md
```

Audio, model weights, credentials, consent documents, and personal data must never be committed to Git. The initial `.gitignore` blocks the most common local data and model artifacts.

## Data requirements

A few songs can be enough for experimentation, but authorization and clean data matter more than raw song count. Each reference must have:

- a stable internal asset identifier;
- proof of permission for the intended training and output use;
- master and composition ownership information;
- performer and voice authorization where applicable;
- approved territories, channels, and commercial-use terms;
- an expiration and revocation policy; and
- complete credit and attribution data.

Do not use material obtained through stream ripping, leaked sessions, private archives without permission, or recordings whose ownership cannot be verified.

## Safety and release requirements

Before a generated track can leave a development environment, the project should require:

- approval from the designated artist or estate representative;
- review for substantial similarity, memorized passages, and accidental lyric reproduction;
- separate clearance for lyrics, compositions, samples, and synthetic vocals;
- a visible `AI-assisted` disclosure wherever the track is presented;
- embedded provenance and a durable project identifier;
- accurate credits for human and AI-assisted contributions; and
- a documented takedown and revocation process.

The project must not be used to impersonate a living or deceased person, deceive listeners, evade music rights, fabricate endorsements, create defamatory material, or publish an alleged “lost recording.”

## Roadmap

- [ ] Define the rights-manifest schema and validation rules.
- [ ] Build a local audio-ingestion and dataset-validation CLI.
- [ ] Establish a reproducible instrumental-generation baseline.
- [ ] Add project-scoped adapter training for small authorized datasets.
- [ ] Create similarity, memorization, and data-leakage evaluations.
- [ ] Add human review, approval, and provenance records.
- [ ] Gate synthetic-vocal support behind explicit voice authorization.
- [ ] Pilot the complete workflow with a participating artist or estate.

## Contributing

Contributions are welcome once the initial architecture and governance documents are in place. By contributing, you agree not to submit copyrighted audio, model weights trained on unauthorized recordings, private consent documents, credentials, or personal data.

For now, open an issue describing the use case, the rights boundary, and the proposed technical change before starting a large contribution.

## License

No open-source license has been selected yet. Until one is added, all rights are reserved and the repository contents may not be reused, modified, or redistributed without permission.

Music, recordings, artist identities, and generated outputs are not automatically covered by any future source-code license; they remain subject to their own rights and agreements.
