# Troubleshooting

Run the lightweight diagnostic first:

```bash
uv run legacy-music doctor
```

The command reports Python, Git, uv, FFmpeg, FFprobe, NVIDIA tooling, and Conda availability without downloading or changing anything.

Model-stage failures will retain the complete run directory and record the isolated environment, reviewed upstream revision, sanitized command, exit code, timestamps, stdout, stderr, and expected output paths. Do not delete failed runs before diagnosis. CUDA out-of-memory handling should recommend reducing batch size, enabling offloading, or selecting a smaller model mode; it must not silently change training semantics.

Never place credentials, contracts, artist recordings, or raw lyrics in issue reports or public logs.
