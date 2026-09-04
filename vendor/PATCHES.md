# Vendored faster-qwen3-tts

Upstream: https://github.com/andimarafioti/faster-qwen3-tts
Version: 0.4.0
Commit: e2a215f61984c0e72a242f8dd72333338e7672f4
License: MIT — see `LICENSE.faster-qwen3-tts` (copyright Andres Marafioti, retained verbatim)

Bundled so this node installs with no pip step. `nodes.py` prefers a
pip-installed `faster_qwen3_tts` if the user has one; this copy is the fallback.

## Files included

`model.py`, `generate.py`, `talker_graph.py`, `predictor_graph.py`,
`sampling.py`, `streaming.py`, `utils.py`, plus a trimmed `__init__.py`.

Not included: `cli.py` (a standalone command-line tool, irrelevant inside
ComfyUI) and `ggml_backend.py` (the optional qwentts.cpp backend, which needs
the separate `qwentts-cpp-python` native wheel).

## Changes applied

Every change is marked in the source with `[ComfyUI vendor patch]`.

### 1. transformers 4.x compatible cache initialisation
`talker_graph.py`, `predictor_graph.py` — in `_init_cache_layers`

Upstream calls `layer.lazy_initialization(dummy_k, dummy_k)`, the transformers
5.x signature. transformers 4.5x takes a single key tensor and derives the value
buffer from its shape. Upstream passes the same tensor twice, so the allocated
cache is identical either way. Wrapped in `try/except TypeError` with the
single-argument call as fallback, so the code runs on both major versions.

This is the only functional difference from upstream.

### 2. GGML backend removed
`model.py` — in `from_pretrained`

`backend="ggml"` / `"qwentts"` now raises a clear `RuntimeError` pointing at the
upstream package instead of importing the module that is not vendored. The
default `backend="torch"` path is untouched.

### 3. Trimmed `__init__.py`
Exports `FasterQwen3TTS` only; the upstream file also exports `GGMLQwen3TTS`.
Adds `VENDORED_UPSTREAM_VERSION` and `VENDORED_UPSTREAM_COMMIT`.

## Syncing with upstream

1. `git clone https://github.com/andimarafioti/faster-qwen3-tts`
2. Copy the seven files above over `vendor/faster_qwen3_tts/`
3. Re-apply the three changes (grep upstream for `lazy_initialization` and
   `ggml_backend` — both are small and localised)
4. Update the version and commit in this file and in `vendor/faster_qwen3_tts/__init__.py`
5. Run a generation before committing
