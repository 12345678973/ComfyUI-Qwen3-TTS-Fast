"""
Bundled copy of faster-qwen3-tts.

Upstream: https://github.com/andimarafioti/faster-qwen3-tts
Copyright (c) 2026 Andres Marafioti -- MIT License (see ../LICENSE.faster-qwen3-tts)

Vendored so this ComfyUI node installs with no pip step. See ../PATCHES.md for
the small set of changes applied and why. `nodes.py` prefers a pip-installed
`faster_qwen3_tts` when one is present; this copy is the fallback.

The optional GGML/qwentts.cpp backend and the CLI are not included.
"""

from .model import FasterQwen3TTS

VENDORED_UPSTREAM_VERSION = "0.4.0"
VENDORED_UPSTREAM_COMMIT = "e2a215f61984c0e72a242f8dd72333338e7672f4"

__all__ = ["FasterQwen3TTS", "VENDORED_UPSTREAM_VERSION", "VENDORED_UPSTREAM_COMMIT"]
