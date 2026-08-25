# Artist profiles

An `ArtistProfile` is the fundamental unit of isolation. It selects one music adapter, one voice engine, one rights manifest, one dataset root, one preset collection, and one evaluation history.

Planned CLI commands:

```text
legacy-music artist create
legacy-music artist list
legacy-music artist status <artist-id>
```

Creating a profile will validate a lowercase hyphenated ID, write files atomically, initialize the expected private directories, and leave all permissions disabled until a rights manifest is approved. Adding a second artist will require no application-code change.

The committed `artists/_template/` is configuration documentation only. Real profiles are ignored by Git.
