# Voice conversion

SoulX-Singer SVC is the planned default `VoiceEngine`; Seed-VC is an optional implementation. Each engine runs in an isolated environment behind the same typed request and output contract.

An artist maintains several short, curated references such as neutral, soft, strong, high-register, and low-register. The MVP maps `auto` to the artist's configured default. Later selection may consider range, energy, tempo, and genre.

Voice references must be approved, clean, single-singer material. Reject significant backing vocals, instrumental bleed, clipping, dialogue, noise, or distortion. Every conversion records the reference ID and hash without exposing private reference audio.
