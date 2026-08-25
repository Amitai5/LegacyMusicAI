# Artist profiles

An `ArtistProfile` is the fundamental unit of isolation. It selects one music adapter, one voice engine, one rights manifest, one dataset root, one preset collection, and one evaluation history.

Implemented CLI commands:

```text
legacy-music artist create
legacy-music artist list
legacy-music artist status <artist-id>
```

Creating a profile validates a lowercase hyphenated ID, writes files atomically, initializes the private directory tree, and leaves every permission disabled until an authorized operator edits the rights manifest. Adding a second artist requires no application-code change.

`artist status` reports each permission independently, catalog size, selected-adapter readiness, and curated voice-reference count. Artist paths are confined to one direct child of `artists/`; symlink and parent traversal are rejected.

The committed `artists/_template/` is configuration documentation only. Real profiles are ignored by Git.
