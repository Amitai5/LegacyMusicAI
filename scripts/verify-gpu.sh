#!/usr/bin/env bash
set -euo pipefail

missing=0
for executable in git uv ffmpeg ffprobe nvidia-smi conda; do
    if command -v "$executable" >/dev/null 2>&1; then
        printf 'ready: %s (%s)\n' "$executable" "$(command -v "$executable")"
    else
        printf 'missing: %s\n' "$executable" >&2
        missing=1
    fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

exit "$missing"
