# Troubleshooting

Run the strict diagnostic first:

```bash
uv run legacy-music doctor --strict --root .
```

The command reports Python, Git, uv, optional FFmpeg/FFprobe, NVIDIA tooling, Conda, and every required ACE/SoulX weight without downloading or changing anything.

Generation failures retain the run directory, last durable stage, prior artifact hashes, and an ACE startup log when the managed service was used. Do not delete failed runs before diagnosis. On an 8 GB GPU, keep the default managed ACE mode for voice generation; an external ACE server can retain enough VRAM to make SoulX fail. Reduce training rank or increase gradient accumulation explicitly rather than silently changing semantics.

Never place credentials, contracts, artist recordings, or raw lyrics in issue reports or public logs.
