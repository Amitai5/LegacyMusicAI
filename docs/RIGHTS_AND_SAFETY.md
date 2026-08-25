# Rights and safety

## Required authorization

Before any real recording is ingested, the operator must verify permission for the exact intended use from every applicable holder. Depending on the project, this can include the artist or estate, master-recording owner, composition publisher, performers, featured vocalists, name/image/likeness holder, and voice-rights holder.

Public access, ownership of a physical copy, a streaming subscription, or an artist's death does not grant training or imitation rights.

## Machine-readable gate

Each artist profile contains a local `rights.yaml` that records:

- lifecycle status: pending, approved, expired, or revoked;
- effective and expiration instants;
- separate training, musical-style, singing-voice, and commercial-release permissions;
- non-secret references to approval evidence;
- authorized source-asset hashes; and
- release controls and disclosure requirements.

Private contracts and identity documents stay outside Git. The manifest stores references, not the evidence itself.

Every protected operation must fail closed when the manifest is missing, invalid, inactive, or does not authorize the requested capability. Voice authorization is separate from authorization to learn musical style.

## Misuse controls

The application must not:

- impersonate an artist without authorization;
- present generated audio as a historical, recovered, leaked, or authentic performance;
- fabricate an endorsement or statement;
- generate defamatory, fraudulent, harassing, or deceptive material;
- bypass master, composition, lyric, sample, performance, voice, or publicity rights;
- train on stream-ripped, leaked, or unverifiable recordings;
- automatically publish, distribute, or monetize an output; or
- conceal AI assistance or provenance.

## Release gate

An export intended to leave the development environment requires:

1. an active authorization manifest covering the output use;
2. human approval from the designated artist or estate representative;
3. checks for memorized passages, substantial source similarity, lyric reproduction, and unapproved samples;
4. review of pronunciation, defamatory implications, and misleading context;
5. accurate human and AI contribution credits;
6. visible AI-assisted disclosure plus durable provenance; and
7. a documented revocation and takedown route.

Passing automated checks is not a substitute for legal clearance or human approval.

## Testing policy

Unit and general integration tests use synthetic or explicitly licensed fixtures. Real-artist evaluations are local, opt-in, access-controlled, and excluded from CI and Git. Logs must not include lyrics, contracts, credentials, personal data, or raw audio content.
