"""
ComfyUI-Qwen3-TTS-Fast

CUDA-graph accelerated Qwen3-TTS nodes, built on faster-qwen3-tts.
Runs in ComfyUI's own Python. The original Qwen3_TTS node pack is untouched
and both can be installed side by side.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
