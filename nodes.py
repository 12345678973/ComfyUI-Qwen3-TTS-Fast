"""
ComfyUI-Qwen3-TTS-Fast

Qwen3-TTS nodes driven by faster-qwen3-tts, which captures the whole decode
step (28-layer talker + 5-layer code predictor) into a replayed CUDA graph
backed by a static KV cache. Reported gains: ~5.6x on RTX 4090, ~7.9x on
RTX 4060 (Windows), ~7.1x on H100.

Node layout follows the crt-nodes Qwen3_TTS wrapper so workflows translate
one-to-one.
"""

import gc
import hashlib
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

import folder_paths

# qwen_tts carries the Qwen3-TTS model definition and tokenizer. It is declared
# in requirements.txt, so ComfyUI-Manager installs it; this probe only exists to
# turn a manual clone without dependencies into a readable message.
try:
    import qwen_tts  # noqa: F401
    QWEN_TTS_AVAILABLE = True
except ImportError:
    QWEN_TTS_AVAILABLE = False
    print("\n" + "=" * 70)
    print("[ComfyUI-Qwen3-TTS-Fast] The 'qwen-tts' package is missing.")
    print("  Install this node through ComfyUI-Manager, or install it manually:")
    print(f"    \"{sys.executable}\" -m pip install qwen-tts")
    print("=" * 70 + "\n")

# faster-qwen3-tts provides the CUDA graph inference path. A pip-installed copy
# wins if the user has one; otherwise the bundled copy in vendor/ is used, so
# this node needs no pip step. See vendor/PATCHES.md.
try:
    from faster_qwen3_tts import FasterQwen3TTS
    FAST_SOURCE = "installed"
    FAST_AVAILABLE = True
except ImportError:
    try:
        from .vendor.faster_qwen3_tts import FasterQwen3TTS
        FAST_SOURCE = "bundled"
        FAST_AVAILABLE = True
    except ImportError as _exc:
        FAST_SOURCE = None
        FAST_AVAILABLE = False
        print("\n" + "=" * 70)
        print("[ComfyUI-Qwen3-TTS-Fast] Could not load the inference backend.")
        print(f"  {_exc}")
        print("  The bundled copy lives in vendor/faster_qwen3_tts/. If it is")
        print("  missing, re-clone the repository. As a fallback you can install")
        print("  upstream:  pip install faster-qwen3-tts --no-deps")
        print("=" * 70 + "\n")

LOAD_ERROR = (
    "The Qwen3-TTS backend could not be loaded.\n"
    "Check the ComfyUI console output from startup for the reason."
)

QWEN_TTS_ERROR = (
    "The 'qwen-tts' package is not installed, so the Qwen3-TTS model "
    "definition is unavailable.\n\n"
    "Install this node through ComfyUI-Manager, which handles it, or run:\n"
    "  \"{exe}\" -m pip install qwen-tts"
).format(exe=sys.executable)

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


MODEL_NAMES = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

REF_CACHE_DIR = os.path.join(tempfile.gettempdir(), "comfy_qwen3_tts_fast_refs")
os.makedirs(REF_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# transformers 4.x / 5.x compatibility
# ---------------------------------------------------------------------------

def _preinit_static_cache(static_cache, config, dtype, device):
    """
    Allocate the StaticCache layer buffers ahead of graph capture.

    faster-qwen3-tts calls layer.lazy_initialization(k, v), which is the
    transformers 5.x signature. transformers 4.57.x takes a single key tensor
    and derives the value buffer from its shape -- identical result, since the
    library passes the same tensor twice. Doing it here sets is_initialized,
    so the library's own loop becomes a no-op on either version.
    """
    num_kv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    dummy = torch.zeros(1, num_kv, 1, head_dim, dtype=dtype, device=device)
    for layer in static_cache.layers:
        if getattr(layer, "is_initialized", False):
            continue
        try:
            layer.lazy_initialization(dummy, dummy)   # transformers 5.x
        except TypeError:
            layer.lazy_initialization(dummy)          # transformers 4.x


def _apply_compat(fast_model):
    tg = fast_model.talker_graph
    _preinit_static_cache(tg.static_cache, tg.model.config, tg.dtype, tg.device)
    pg = fast_model.predictor_graph
    _preinit_static_cache(pg.static_cache, pg.pred_model.config, pg.dtype, pg.device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    if seed is None or seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def _audio_to_ref_path(audio):
    """
    ComfyUI AUDIO -> a mono WAV on disk, named by content hash.

    faster-qwen3-tts keys its voice-prompt cache on the file path, so a stable
    hash means repeat generations with the same reference skip speaker
    embedding extraction entirely.
    """
    wav = audio["waveform"]
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    wav = np.asarray(wav)
    while wav.ndim > 2:
        wav = wav[0]
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    sr = int(audio["sample_rate"])

    digest = hashlib.sha1(wav.tobytes() + str(sr).encode()).hexdigest()[:20]
    path = os.path.join(REF_CACHE_DIR, f"{digest}.wav")
    if not os.path.exists(path):
        sf.write(path, wav, sr, subtype="FLOAT")
    return path, wav, sr


def _wav_to_audio(wavs, sr):
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    waveform = torch.from_numpy(np.ascontiguousarray(wav))
    return {"waveform": waveform.unsqueeze(0).unsqueeze(0), "sample_rate": int(sr)}


def _clamp_tokens(model, max_new_tokens):
    limit = max(64, model.max_seq_len - 256)
    if max_new_tokens > limit:
        print(f"[Qwen3-TTS Fast] max_new_tokens {max_new_tokens} exceeds the static cache "
              f"budget; clamping to {limit}. Raise max_seq_len on the loader to go higher.")
        return limit
    return max_new_tokens


def _sampling(model, max_new_tokens, temperature, top_p, repetition_penalty):
    return dict(
        max_new_tokens=_clamp_tokens(model, int(max_new_tokens)),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )


def _timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_vram = torch.cuda.memory_allocated()
    else:
        start_vram = 0
    t0 = time.time()
    wavs, sr = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_vram = torch.cuda.memory_allocated()
    else:
        end_vram = 0
    elapsed = time.time() - t0

    total = sum(np.asarray(w).reshape(-1).shape[0] for w in wavs) if isinstance(wavs, (list, tuple)) else len(wavs)
    duration = total / float(sr)
    rtf = duration / elapsed if elapsed > 0 else 0.0

    print("\n[Qwen3-TTS Fast Stats]")
    print(f"  Speed: {rtf:.2f}x Real-Time ({duration:.2f}s audio in {elapsed:.2f}s)")
    if torch.cuda.is_available():
        print(f"  VRAM:  {start_vram / 1024**3:.2f} GB -> {end_vram / 1024**3:.2f} GB")
    print("-" * 30)

    return wavs, sr


def _check(model):
    if not isinstance(model, FastTTSModel):
        raise RuntimeError("Connect the 'Qwen3 TTS Fast Loader' node to the model input.")
    return model


# ---------------------------------------------------------------------------
# Model handle
# ---------------------------------------------------------------------------

class FastTTSModel:
    """Wraps FasterQwen3TTS with the bits the nodes need."""

    def __init__(self, fast, model_name, max_seq_len):
        self.fast = fast
        self.base = fast.model          # the underlying qwen_tts.Qwen3TTSModel
        self.model_name = model_name
        self.max_seq_len = max_seq_len

    def create_clone_prompt(self, wav, sr, ref_text, x_vector_only):
        if x_vector_only:
            return self.base.create_voice_clone_prompt(
                ref_audio=(wav, sr), ref_text="", x_vector_only_mode=True
            )
        # Trailing silence keeps the reference's last phoneme out of the first
        # generated token -- what the library does on its own file path.
        padded = np.concatenate([wav, np.zeros(int(0.5 * sr), dtype=np.float32)])
        return self.base.create_voice_clone_prompt(ref_audio=(padded, sr), ref_text=ref_text)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class Qwen3TTSFastLoader:
    """Load a Qwen3-TTS model and capture its CUDA graphs."""

    _models = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_NAMES,),
                "device": (["cuda:0", "cuda:1"],),
                "dtype": (["bfloat16", "float16"],),
                "attention": (["sdpa", "flash_attention_2"],),
                "max_seq_len": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 128}),
            },
            "optional": {
                "warmup": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("QWEN3TTS_FAST",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3_TTS_Fast"

    def load_model(self, model_name, device, dtype, attention, max_seq_len, warmup=True):
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError(QWEN_TTS_ERROR)
        if not FAST_AVAILABLE:
            raise RuntimeError(LOAD_ERROR)
        if not torch.cuda.is_available():
            raise RuntimeError("faster-qwen3-tts needs CUDA. No GPU visible to this ComfyUI.")

        tts_dir = os.path.join(folder_paths.models_dir, "TTS")
        local_path = os.path.join(tts_dir, model_name.split("/")[-1])

        if not os.path.isdir(local_path) or not os.listdir(local_path):
            if HF_HUB_AVAILABLE:
                print(f"[Qwen3-TTS Fast] Downloading {model_name} -> {local_path}")
                snapshot_download(repo_id=model_name, local_dir=local_path)
            else:
                print("[Qwen3-TTS Fast] huggingface_hub missing; using the default HF cache.")

        load_path = local_path if os.path.isdir(local_path) and os.listdir(local_path) else model_name

        key = (load_path, device, dtype, attention, int(max_seq_len))
        if key in self._models:
            return (self._models[key],)

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

        print(f"[Qwen3-TTS Fast] Loading {load_path} ({dtype}, {attention}, "
              f"max_seq_len={max_seq_len}) via {FAST_SOURCE} faster-qwen3-tts")
        fast = FasterQwen3TTS.from_pretrained(
            load_path,
            device=device,
            dtype=torch_dtype,
            attn_implementation=attention,
            max_seq_len=int(max_seq_len),
        )

        _apply_compat(fast)

        if warmup:
            print("[Qwen3-TTS Fast] Capturing CUDA graphs...")
            t0 = time.time()
            fast.warmup()
            print(f"[Qwen3-TTS Fast] Graphs captured in {time.time() - t0:.1f}s")

        handle = FastTTSModel(fast, model_name, int(max_seq_len))
        self._models[key] = handle
        return (handle,)


class Qwen3TTSFastCustomVoice:
    """Built-in speakers (CustomVoice models)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_FAST",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "speaker": (SPEAKERS,),
                "language": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3_TTS_Fast"

    def generate(self, model, text, speaker, language, seed, instruct="",
                 max_new_tokens=2048, temperature=1.0, top_p=0.8, repetition_penalty=1.1):
        m = _check(model)
        set_seed(seed)
        kw = _sampling(m, max_new_tokens, temperature, top_p, repetition_penalty)
        wavs, sr = _timed(lambda: m.fast.generate_custom_voice(
            text=text, speaker=speaker, language=language,
            instruct=instruct or None, **kw))
        return (_wav_to_audio(wavs, sr),)


class Qwen3TTSFastVoiceDesign:
    """Voice from a written description (VoiceDesign models)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_FAST",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "A warm, gentle young female voice with clear pronunciation.",
                }),
                "language": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3_TTS_Fast"

    def generate(self, model, text, voice_description, language, seed,
                 max_new_tokens=2048, temperature=1.0, top_p=0.8, repetition_penalty=1.1):
        m = _check(model)
        set_seed(seed)
        kw = _sampling(m, max_new_tokens, temperature, top_p, repetition_penalty)
        wavs, sr = _timed(lambda: m.fast.generate_voice_design(
            text=text, instruct=voice_description, language=language, **kw))
        return (_wav_to_audio(wavs, sr),)


class Qwen3TTSFastVoiceClone:
    """Clone from an AUDIO input (Base models)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_FAST",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": "Transcript of the reference audio"}),
                "language": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3_TTS_Fast"

    def generate(self, model, text, ref_audio, ref_text, language, seed, x_vector_only=False,
                 max_new_tokens=2048, temperature=1.0, top_p=0.8, repetition_penalty=1.1):
        m = _check(model)
        set_seed(seed)
        path, _, _ = _audio_to_ref_path(ref_audio)
        kw = _sampling(m, max_new_tokens, temperature, top_p, repetition_penalty)
        wavs, sr = _timed(lambda: m.fast.generate_voice_clone(
            text=text, language=language, ref_audio=path,
            ref_text="" if x_vector_only else ref_text,
            xvec_only=bool(x_vector_only), **kw))
        return (_wav_to_audio(wavs, sr),)


class Qwen3TTSFastVoiceCloneFromFile:
    """Clone from a WAV path on disk (Base models)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_FAST",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "ref_audio_path": ("STRING", {"default": "C:/path/to/audio.wav"}),
                "ref_text": ("STRING", {"multiline": True, "default": "Transcript of the reference audio"}),
                "language": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3_TTS_Fast"

    def generate(self, model, text, ref_audio_path, ref_text, language, seed, x_vector_only=False,
                 max_new_tokens=2048, temperature=1.0, top_p=0.8, repetition_penalty=1.1):
        m = _check(model)
        if not os.path.isfile(ref_audio_path):
            raise RuntimeError(f"Reference audio not found: {ref_audio_path}")
        set_seed(seed)
        kw = _sampling(m, max_new_tokens, temperature, top_p, repetition_penalty)
        wavs, sr = _timed(lambda: m.fast.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio_path,
            ref_text="" if x_vector_only else ref_text,
            xvec_only=bool(x_vector_only), **kw))
        return (_wav_to_audio(wavs, sr),)


class Qwen3TTSFastCreateClonePrompt:
    """Extract a reusable voice-clone prompt once, then feed it to many lines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_FAST",),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": "Transcript of the reference audio"}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("QWEN3TTS_FAST_PROMPT",)
    RETURN_NAMES = ("clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3_TTS_Fast"

    def create_prompt(self, model, ref_audio, ref_text, x_vector_only=False):
        m = _check(model)
        _, wav, sr = _audio_to_ref_path(ref_audio)
        items = m.create_clone_prompt(wav, sr, "" if x_vector_only else ref_text, x_vector_only)
        return ({"items": items, "ref_text": "" if x_vector_only else ref_text},)


class Qwen3TTSFastCloneWithPrompt:
    """Generate using a prompt from 'Create Clone Prompt (Fast)'."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_FAST",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "clone_prompt": ("QWEN3TTS_FAST_PROMPT",),
                "language": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3_TTS_Fast"

    def generate(self, model, text, clone_prompt, language, seed,
                 max_new_tokens=2048, temperature=1.0, top_p=0.8, repetition_penalty=1.1):
        m = _check(model)
        set_seed(seed)
        kw = _sampling(m, max_new_tokens, temperature, top_p, repetition_penalty)
        wavs, sr = _timed(lambda: m.fast.generate_voice_clone(
            text=text, language=language,
            voice_clone_prompt=clone_prompt["items"],
            ref_text=clone_prompt["ref_text"], **kw))
        return (_wav_to_audio(wavs, sr),)


class Qwen3TTSFastBatchGenerate:
    """One clip per line, generated sequentially on the loaded model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_FAST",),
                "text_list": ("STRING", {"multiline": True, "default": "Line 1\nLine 2\nLine 3"}),
                "separator": ("STRING", {"default": "\n"}),
                "language": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "speaker": (SPEAKERS,),
                "clone_prompt": ("QWEN3TTS_FAST_PROMPT",),
                "increment_seed": ("BOOLEAN", {"default": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_batch"
    CATEGORY = "Qwen3_TTS_Fast"

    def generate_batch(self, model, text_list, separator, language, seed,
                       speaker="Vivian", clone_prompt=None, increment_seed=True,
                       max_new_tokens=2048, temperature=1.0, top_p=0.8, repetition_penalty=1.1):
        m = _check(model)
        sep = separator if separator else "\n"
        texts = [t.strip() for t in text_list.split(sep) if t.strip()]
        if not texts:
            raise RuntimeError("text_list is empty after splitting on the separator.")

        kw = _sampling(m, max_new_tokens, temperature, top_p, repetition_penalty)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        audios = []
        total_samples = 0
        sr = 24000
        for i, txt in enumerate(texts):
            set_seed(int(seed) + (i if increment_seed else 0))
            if clone_prompt:
                wavs, sr = m.fast.generate_voice_clone(
                    text=txt, language=language,
                    voice_clone_prompt=clone_prompt["items"],
                    ref_text=clone_prompt["ref_text"], **kw)
            else:
                wavs, sr = m.fast.generate_custom_voice(
                    text=txt, speaker=speaker, language=language, instruct=None, **kw)
            audio = _wav_to_audio(wavs, sr)
            total_samples += audio["waveform"].shape[-1]
            audios.append(audio)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        duration = total_samples / float(sr)
        rtf = duration / elapsed if elapsed > 0 else 0.0

        print("\n[Qwen3-TTS Fast Batch Stats]")
        print(f"  {len(texts)} clips | {duration:.2f}s audio in {elapsed:.2f}s ({rtf:.2f}x Real-Time)")
        print("-" * 30)

        return (audios,)


class Qwen3TTSFastUnload:
    """
    Drop the model and its CUDA graphs to free VRAM.

    Chain the audio through this node so it runs after generation. The captured
    graphs hold fixed GPU buffers, so the model cannot be offloaded to CPU and
    brought back -- it is fully released and reloaded on the next run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "unload"
    CATEGORY = "Qwen3_TTS_Fast"
    OUTPUT_NODE = False

    def unload(self, audio, enabled=True):
        if enabled:
            Qwen3TTSFastLoader._models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("[Qwen3-TTS Fast] Model released, VRAM freed.")
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "Qwen3TTSFastLoader": Qwen3TTSFastLoader,
    "Qwen3TTSFastCustomVoice": Qwen3TTSFastCustomVoice,
    "Qwen3TTSFastVoiceDesign": Qwen3TTSFastVoiceDesign,
    "Qwen3TTSFastVoiceClone": Qwen3TTSFastVoiceClone,
    "Qwen3TTSFastVoiceCloneFromFile": Qwen3TTSFastVoiceCloneFromFile,
    "Qwen3TTSFastCreateClonePrompt": Qwen3TTSFastCreateClonePrompt,
    "Qwen3TTSFastCloneWithPrompt": Qwen3TTSFastCloneWithPrompt,
    "Qwen3TTSFastBatchGenerate": Qwen3TTSFastBatchGenerate,
    "Qwen3TTSFastUnload": Qwen3TTSFastUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSFastLoader": "Qwen3 TTS Fast Loader",
    "Qwen3TTSFastCustomVoice": "Qwen3 TTS Custom Voice (Fast)",
    "Qwen3TTSFastVoiceDesign": "Qwen3 TTS Voice Design (Fast)",
    "Qwen3TTSFastVoiceClone": "Qwen3 TTS Voice Clone (Fast)",
    "Qwen3TTSFastVoiceCloneFromFile": "Qwen3 TTS Voice Clone File (Fast)",
    "Qwen3TTSFastCreateClonePrompt": "Qwen3 TTS Create Clone Prompt (Fast)",
    "Qwen3TTSFastCloneWithPrompt": "Qwen3 TTS Clone with Prompt (Fast)",
    "Qwen3TTSFastBatchGenerate": "Qwen3 TTS Batch Generate (Fast)",
    "Qwen3TTSFastUnload": "Qwen3 TTS Fast Unload",
}
