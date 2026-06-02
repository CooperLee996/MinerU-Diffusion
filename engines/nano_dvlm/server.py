from __future__ import annotations

import asyncio
import argparse
import base64
import io
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

NANO_DVLM_DIR = Path(__file__).resolve().parent
if str(NANO_DVLM_DIR) not in sys.path:
    sys.path.insert(0, str(NANO_DVLM_DIR))

from common import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    DEFAULT_GEN_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    STOP_STRINGS,
    SYSTEM_PROMPT,
    TASK_PROMPTS,
)
from nanovllm import LLM, SamplingParams  # noqa: E402


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_NAME
    messages: list[dict[str, Any]]
    max_tokens: int = DEFAULT_GEN_LENGTH
    temperature: float = 1.0
    stream: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)


class _EngineState:
    llm: LLM | None = None
    model_path: Path | None = None
    served_model_name: str = DEFAULT_MODEL_NAME


state = _EngineState()


def _load_mask_token_id(model_path: Path) -> int:
    with open(model_path / "config.json", "r", encoding="utf-8") as fh:
        config = json.load(fh)
    mask_token_id = config.get("mask_token_id")
    if mask_token_id is None:
        raise ValueError(f"mask_token_id is missing from {model_path / 'config.json'}")
    return mask_token_id


def _decode_data_url(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("expected a data URL for inline image content")
    _, encoded = url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def _normalize_content_part(part: dict[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "text", "text": part.get("text", "")}
    if part_type == "image_url":
        image_url = part.get("image_url") or {}
        url = image_url.get("url", "")
        if url.startswith("data:"):
            return {"type": "image", "image": _decode_data_url(url)}
        return {"type": "image", "image": url}
    if part_type == "image":
        image = part.get("image")
        if isinstance(image, str) and image.startswith("data:"):
            return {"type": "image", "image": _decode_data_url(image)}
        return {"type": "image", "image": image}
    raise ValueError(f"unsupported message content part type: {part_type}")


def _normalize_openai_messages(messages: list[dict[str, Any]]) -> list[dict]:
    normalized: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")

        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be a string or a list of parts")

        normalized.append(
            {
                "role": role,
                "content": [_normalize_content_part(part) for part in content],
            }
        )
    return normalized


def _apply_prompt_type(messages: list[dict], prompt_type: str | None) -> list[dict]:
    if not prompt_type:
        return messages
    if prompt_type not in TASK_PROMPTS:
        raise ValueError(f"unsupported prompt_type: {prompt_type}")

    suffix = TASK_PROMPTS[prompt_type]
    updated = [dict(message) for message in messages]
    for message in reversed(updated):
        if message.get("role") != "user":
            continue
        content = message["content"]
        if not content or content[-1].get("type") != "text":
            content = list(content) + [{"type": "text", "text": suffix}]
        else:
            content = list(content)
            content[-1] = {
                "type": "text",
                "text": f"{content[-1].get('text', '')}{suffix}",
            }
        message["content"] = content
        break
    return updated


def _ensure_system_prompt(messages: list[dict], include_system_prompt: bool) -> list[dict]:
    if not include_system_prompt:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        *messages,
    ]


def _strip_stop_strings(text: str) -> str:
    for stop in STOP_STRINGS:
        text = text.split(stop, 1)[0]
    return text.strip()


def _build_sampling_params(request: ChatCompletionRequest) -> SamplingParams:
    extra = request.extra_body or {}
    remask_strategy = extra.get("denoising_strategy", "low_confidence_dynamic")
    dynamic_threshold = float(extra.get("dynamic_threshold", 0.95))
    temperature = float(request.temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for diffusion sampling")

    return SamplingParams(
        temperature=temperature,
        max_new_tokens=int(request.max_tokens),
        denoising_strategy=remask_strategy,
        dynamic_threshold=dynamic_threshold,
        stop_tokens=list(STOP_STRINGS),
    )


def _create_llm(args: argparse.Namespace) -> LLM:
    model_path = Path(args.model_path).resolve()
    mask_token_id = _load_mask_token_id(model_path)
    return LLM(
        str(model_path),
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        data_parallel_size=args.data_parallel_size,
        mask_token_id=mask_token_id,
        block_size=args.block_size,
        max_model_len=args.max_length,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    yield
    if state.llm is not None:
        state.llm.exit()
        state.llm = None


app = FastAPI(title="MinerU Nano-DVLM Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if state.llm is not None else "uninitialized",
        "model_path": str(state.model_path) if state.model_path else None,
        "data_parallel_size": state.llm.data_parallel_size if state.llm else 0,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": state.served_model_name,
                "object": "model",
                "owned_by": "mineru-diffusion",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
    if state.llm is None:
        raise HTTPException(status_code=503, detail="engine is not initialized")

    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="stream=true is not supported for block diffusion decoding",
        )

    extra = request.extra_body or {}
    try:
        messages = _normalize_openai_messages(request.messages)
        messages = _ensure_system_prompt(messages, extra.get("include_system_prompt", True))
        messages = _apply_prompt_type(messages, extra.get("prompt_type"))
        sampling_params = _build_sampling_params(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.time()
    loop = asyncio.get_running_loop()
    try:
        outputs = await loop.run_in_executor(
            None,
            lambda: state.llm.generate_messages(
                [messages],  # batch of one conversation
                sampling_params=sampling_params,
                use_tqdm=False,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    text = _strip_stop_strings(outputs[0]["text"])
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(started),
        "model": request.model or state.served_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(outputs[0].get("token_ids", [])),
            "total_tokens": len(outputs[0].get("token_ids", [])),
        },
    }
    return JSONResponse(payload)


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True, help="Converted HF model directory.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")


def run_server(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for nano_dvlm server")

    state.model_path = Path(args.model_path).resolve()
    state.served_model_name = args.served_model_name
    state.llm = _create_llm(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MinerU Nano-DVLM OpenAI-compatible server.")
    add_server_arguments(parser)
    run_server(parser.parse_args())


if __name__ == "__main__":
    main()
