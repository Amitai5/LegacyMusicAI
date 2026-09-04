# Voice conversion

SoulX-Singer SVC is the implemented default `VoiceEngine`; artist-fine-tuned Seed-VC is an optional, human-selected alternative. Both run in isolated Python 3.10 Conda environments behind typed preprocessing and conversion contracts. The quality engines return converted audio, phrase decisions, selected reference IDs, aggregate QC metrics, cleaned F0, and a versioned manifest.

An artist maintains several short references such as neutral, soft, strong, high-register,
and low-register. Whole-song `generate --reference auto` uses the artist default for
compatibility. SoulX quality remixing and identity-gated Seed-VC remixing select references
per phrase. Seed-VC identity mode additionally compares candidates with CAMPPlus embeddings
from the approved bank. Similarity scores are not percentages of identity accuracy.

## References and clean training vocals

`legacy-music voice add-reference` requires active singing-voice permission and an exact source hash in `rights.yaml`. It copies the source, verifies the copy, creates a 24 kHz mono derivative, extracts F0, and persists source/audio/F0 hashes. Generation accepts only a curated reference ID; arbitrary paths are rejected.

`voice clean-vocals` processes every verified vocal stem with the reviewed dereverberation model in one sequential GPU batch. It never changes `vocals.wav`; it creates a sibling `clean_vocals.wav`, blends 90% dereverberated audio with 10% of the original stem to retain warmth, matches active level, protects the −1 dBFS peak, rejects clipping/non-finite samples or timing changes, and additively records source/model/config/output hashes in `stems.json` and a batch manifest.

`voice build-bank` derives five 15-second roles (`neutral`, `soft`, `strong`, `low`, and `high`) only from authorized `clean_vocals.wav` studio stems. Candidates must contain 70–95% voiced content, have stable active levels, and contain no clipped samples. Concert/live recordings and objectively poor separation candidates are excluded. Legacy wet references remain loadable; approved clean replacements receive versioned IDs.

`voice prepare-training` creates 1–30-second, mono PCM24 training excerpts only from hash-verified `clean_vocals.wav` files that pass lead-vocal gates for voiced density, continuity, level stability, spectral content, and center dominance. Dataset-policy ranges remove reviewed background vocals, distant fragments, ad-libs, and bleed; an operator can exclude an entire source when safe lead isolation is not possible. Every segment retains clean-stem and original-source lineage, and held-out segments are kept for review when enough candidates exist. `voice train-model` then fine-tunes the pinned Seed-VC 44.1 kHz singing preset in its isolated environment. A completed checkpoint is selected only with `--select`; the default generation engine remains SoulX pending human A/B approval.

These filters are heuristics, not a reliable semantic classifier for lead vocals or a
guarantee of dry, studio-quality audio. Listen to retained bank/training excerpts and record
missed contamination in the private dataset policy. Quality approval of a reference does
not approve release of generated songs. See [dataset policy](DATASET.md) and the separate
[dereverb installation step](../environments/soulx/README.md).

## Immutable quality-first remixes

`voice remix-run <run-id> --reference auto` creates an immutable child of a complete run.
It verifies current rights, parent inputs, and available recorded hashes, then copies the
exact draft, stems, F0, and lyrics. `--duration 30` instead copies a prefix for a smoke test.
Review this short result before a full run; the CLI does not enforce the preliminary review.

The SoulX wrapper loads its model once, disables global pitch shifting, repairs F0 gaps no
longer than 120 ms and isolated octave errors, preserves longer rests, selects a prompt per
12–20-second phrase, generates two seeded candidates, and uses one-second equal-power crossfades.

`--engine seed-vc` uses the selected clean-data fine-tune with F0 conditioning, no automatic
pitch shift, and independent post-conversion F0 QC. Without `--identity-gated`, it converts
the whole song. Add `--identity-gated` for:

- overlapping 6–8-second phrases and five initial seeded candidates using approved reference
  roles ranked by register and energy, with inference guidance set to `1.0`;
- rejection of identity, pitch, dropout, clipping, or spectral failures;
- overlapping four-second active-vocal identity windows, prioritizing the weakest window;
- a bank-calibrated local floor that supersedes silence-biased whole-phrase scoring, while
  retaining whole-phrase scores for audit;
- adaptive retries through complementary continuity-safe and identity-strong references when
  fewer than two initial candidates pass, capped at 15 candidates per phrase; and
- recorded same-candidate repair of low-level waveform holes up to 120 ms, using a smoothed
  envelope capped at 12 dB, followed by rescoring. True silence or longer gaps still fail.

Post-conversion QC separately repairs detector gaps up to 120 ms before pitch scoring.
Seed-VC, its pitch model, and its identity encoder are loaded once per run; vendor code
remains unchanged. Seeds and decisions are recorded, but GPU output is not guaranteed
bit-identical across devices or fresh model processes.

## Independent final mixing and review

The synchronized output path matches the converted stem to the guide globally, applies smoothed local automation capped at ±6 dB, uses light 2:1 compression with 15 ms attack and 120 ms release, and restores controlled mix-stage reverb independently of the dry training data. A content-aware balance pass measures active lead-vocal frames and attenuates only the louder derived bus toward a configurable +1.5 dB target, capped at 12 dB; `--vocal-balance-target-db` provides a per-song override. It then masters through locked FFmpeg at −14 LUFS, LRA 9, and −1 dBTP. Every conversion records all prompt audio/F0/source/stem hashes without exposing private paths. Candidate phrases, leveled vocals, both buses, pre/post balance, automatic correction, premaster, mastering measurements, QC, final output, and provenance are retained. Automatic distribution remains disabled and human review is required.

References must be approved, clean, single-singer material. Reject significant backing vocals, instrumental bleed, clipping, dialogue, noise, or distortion. Identity-gated conversion costs roughly one inference per candidate per phrase, so shorter phrases and five candidates take materially longer than whole-song conversion. It remains opt-in until a human approves the full-song A/B.

The additive voice-conversion manifest is version 4 and retains compatibility with supported
older manifests/references. Passing its automated gates does not guarantee a perfect voice
match, correct pronunciation, or recovery of missing notes in the guide performance.
Outputs stay private, AI-disclosed, and human-review gated.
