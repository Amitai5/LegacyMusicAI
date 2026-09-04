# Run storage

Generation runs are immutable runtime artifacts and are ignored by Git. Every generation receives a unique directory containing its request, durable state, logs, provenance, intermediates, and outputs. Artist training output remains under the corresponding private artist root.

`legacy-music voice remix-run <run-id> --reference auto` creates a child of a completed
run, verifying and reusing its draft, lyrics, stems, and recorded hashes. Full-length remixes
copy those inputs byte-for-byte; `--duration 30` creates a prefix-only smoke-test child.
Neither path overwrites the parent. Optional Seed-VC identity gating uses
`--engine seed-vc --identity-gated`.

Child runs retain candidate phrases, conversion decisions, leveled vocals, mix buses,
mastering measurements, QC, `output/final.wav`, and `output/provenance.json`. Provenance
starts with `release_approved: false`; successful automated QC does not approve distribution.
Do not commit these outputs or attach private manifests and logs to public issues.

Inspect local history with `legacy-music runs list` and `legacy-music runs show <run-id>`.
Failed runs are retained for diagnosis. General resume and reproduction commands remain
backlog work; a vocal remix is not a resume or a bit-identical model replay.
