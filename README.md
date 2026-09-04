# ComfyUI-Qwen3-TTS-Fast

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) nodes for ComfyUI running on
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), which
captures the decode step into a replayed CUDA graph instead of dispatching
~500 kernels per token from Python.

Upstream benchmarks: 5.6x on RTX 4090, 7.1x on H100, 7.9-9.8x on RTX 4060
(Windows). The faster your GPU relative to your CPU, the bigger the gain.

## Install

ComfyUI-Manager → search Qwen3-TTS → install → restart. Nothing else to do.

Manual:

```
cd ComfyUI/custom_nodes
git clone https://github.com/YildirimMC/ComfyUI-Qwen3-TTS-Fast
pip install -r ComfyUI-Qwen3-TTS-Fast/requirements.txt
```

Needs an NVIDIA GPU and PyTorch 2.5.1+. Models download to
`ComfyUI/models/TTS/` on first use.

`faster-qwen3-tts` is bundled in `vendor/` rather than installed, because its
metadata declares `transformers>=5.15` while the code imports one symbol from
transformers. Installing it normally would upgrade transformers across your
ComfyUI. If you'd rather track upstream, `pip install faster-qwen3-tts --no-deps`
— an installed copy takes precedence, and the console says which is in use.

## Nodes

Under the `Qwen3_TTS_Fast` category.

| Node | Purpose |
|---|---|
| Fast Loader | Load a model, capture CUDA graphs |
| Custom Voice (Fast) | Built-in speakers (CustomVoice models) |
| Voice Design (Fast) | Voice from a written description (VoiceDesign models) |
| Voice Clone (Fast) | Clone from an AUDIO input (Base models) |
| Voice Clone File (Fast) | Clone from a WAV path |
| Create Clone Prompt (Fast) | Extract a reusable clone prompt once |
| Clone with Prompt (Fast) | Generate from that prompt |
| Batch Generate (Fast) | One clip per line |
| Fast Unload | Release the model, free VRAM |

Models: `Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` and `-CustomVoice`,
`1.7B-VoiceDesign`. Speed prints to the console as `Nx Real-Time`.

**Loader settings.** `max_seq_len` sizes the static KV cache — longer text needs
more, and `max_new_tokens` above `max_seq_len - 256` is clamped with a warning.
In ICL cloning the reference audio counts against it too. `attention` is `sdpa`
unless you have flash-attn. Leave `warmup` on.

## Limitations

- Output is not bit-identical to other Qwen3-TTS wrappers. The static-cache path
  uses different SDPA kernels, and the code predictor's sampling is fixed at
  graph-capture time, so `temperature` and `top_p` steer the talker stage only.
- No CPU offload — captured graphs hold fixed GPU buffers. The Unload node
  releases the model fully; it reloads on the next run.
- One generation at a time.

## transformers 4.x and 5.x

`faster-qwen3-tts` calls `lazy_initialization(key, value)`, the transformers 5.x
signature. 4.57.x takes only the key and derives the value buffer from its
shape, and upstream passes the same tensor twice, so the caches are identical.
The bundled copy tries both. Details in [vendor/PATCHES.md](vendor/PATCHES.md).

## Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Alibaba Cloud, Qwen Team (Apache-2.0)
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) — Andres Marafioti (MIT), bundled in `vendor/`
- [ComfyUI-QWEN3_TTS](https://github.com/PGCRT/ComfyUI-QWEN3_TTS) — PGCRT (Apache-2.0), whose node layout this follows so workflows translate

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
