#!/usr/bin/env python3
"""Send one OpenAI-style OCR request to a Nano-DVLM server."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import DEFAULT_MODEL_NAME, STOP_STRINGS, SYSTEM_PROMPT, TASK_PROMPTS


def _build_payload(
    model: str,
    prompt: str,
    image_path: Path,
    max_tokens: int,
    temperature: float,
    prompt_type: str | None,
) -> bytes:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    suffix = image_path.suffix.lstrip(".").lower() or "png"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{suffix};base64,{image_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": {
            "denoising_strategy": "low_confidence_dynamic",
            "dynamic_threshold": 0.95,
        },
    }
    if prompt_type is not None:
        payload["extra_body"]["prompt_type"] = prompt_type
    return json.dumps(payload).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one request to Nano-DVLM server.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--prompt-type", choices=sorted(TASK_PROMPTS.keys()), default="text")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    prompt = args.prompt or TASK_PROMPTS[args.prompt_type]
    image_path = Path(args.image_path).resolve()
    payload = _build_payload(
        args.model,
        prompt,
        image_path,
        args.max_tokens,
        args.temperature,
        None if args.prompt else args.prompt_type,
    )

    request = urllib.request.Request(
        args.server_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc

    content = body["choices"][0]["message"]["content"]
    for stop in STOP_STRINGS:
        content = content.split(stop, 1)[0]
    print(content.strip())


if __name__ == "__main__":
    main()
