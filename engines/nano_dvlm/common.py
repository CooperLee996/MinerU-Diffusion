from __future__ import annotations

STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")
SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL_NAME = "mineru-diffusion"
TASK_PROMPTS = {
    "text": "\nText Recognition:",
    "table": "\nTable Recognition:",
    "formula": "\nFormula Recognition:",
    "layout": "\nLayout Analysis:",
}
DEFAULT_MAX_LENGTH = 4096
DEFAULT_GEN_LENGTH = 1024
DEFAULT_BLOCK_SIZE = 32
