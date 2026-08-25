# Voice conversion

SoulX-Singer SVC is the implemented default `VoiceEngine`; Seed-VC is an optional, disabled alternative. SoulX runs in its own Python 3.10 Conda environment behind typed preprocessing and conversion contracts.

An artist maintains several short, curated references such as neutral, soft, strong, high-register, and low-register. The MVP maps `auto` to the artist's configured default. Later selection may consider range, energy, tempo, and genre.

`legacy-music voice add-reference` requires active singing-voice permission and an exact source hash in `rights.yaml`. It copies the source, verifies the copy, creates a 24 kHz mono derivative, extracts F0, and persists source/audio/F0 hashes. Generation accepts only a curated reference ID; arbitrary paths are rejected.

The synchronized path separates an ACE-Step draft into lead vocal and accompaniment, extracts target F0, converts the vocal, and remixes without changing its duration. Every conversion records the reference ID, authorized source hash, normalized-audio hash, and F0 hash without exposing private audio.

References must be approved, clean, single-singer material. Reject significant backing vocals, instrumental bleed, clipping, dialogue, noise, or distortion. Selection beyond the configured default remains a human decision.
