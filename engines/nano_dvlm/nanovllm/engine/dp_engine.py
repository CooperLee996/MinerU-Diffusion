from __future__ import annotations

import atexit
import threading
import traceback
from dataclasses import asdict, fields
from itertools import count

import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


_DIST_PORT_BASE = 24000
_STOP = "__nanovllm_dp_stop__"


def _sampling_params_to_dict(sampling_params: SamplingParams) -> dict:
    return asdict(sampling_params)


def _sampling_params_from_dict(payload: dict) -> SamplingParams:
    allowed = {field.name for field in fields(SamplingParams)}
    return SamplingParams(**{key: value for key, value in payload.items() if key in allowed})


def _dp_worker_main(
    model: str,
    gpu_id: int,
    config_kwargs: dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    worker_kwargs = dict(config_kwargs)
    worker_kwargs["device_id"] = gpu_id
    worker_kwargs["dist_port"] = _DIST_PORT_BASE + gpu_id
    worker_kwargs["tensor_parallel_size"] = 1

    engine = LLMEngine(model, **worker_kwargs)
    try:
        while True:
            item = task_queue.get()
            if item is None or item == _STOP:
                break
            req_id, messages, sampling_params_dict = item
            try:
                sampling_params = _sampling_params_from_dict(sampling_params_dict)
                outputs = engine.generate_messages(
                    [messages],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                result_queue.put((req_id, outputs, None))
            except Exception:
                result_queue.put((req_id, None, traceback.format_exc()))
    finally:
        engine.exit()


class DPLLMEngine:
    """Data-parallel coordinator: one full model replica per GPU."""

    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {key: value for key, value in kwargs.items() if key in config_fields}
        self.model = model
        self.config_kwargs = config_kwargs

        data_parallel_size = kwargs.get("data_parallel_size")
        if data_parallel_size is None:
            data_parallel_size = kwargs.get("dp_size", 1)
        self.data_parallel_size = int(data_parallel_size)
        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")

        tensor_parallel_size = int(config_kwargs.get("tensor_parallel_size", 1))
        if self.data_parallel_size > 1 and tensor_parallel_size != 1:
            raise ValueError(
                "data_parallel_size > 1 requires tensor_parallel_size=1; "
                "use either DP or TP, not both."
            )
        if self.data_parallel_size > torch.cuda.device_count():
            raise ValueError(
                f"data_parallel_size={self.data_parallel_size} exceeds visible CUDA devices "
                f"({torch.cuda.device_count()})."
            )

        self._request_counter = count()
        self._round_robin = 0
        self._pending: dict[int, dict] = {}
        self._pending_lock = threading.Lock()
        self._closed = False

        ctx = mp.get_context("spawn")
        self._result_queue = ctx.Queue()
        self._task_queues: list[mp.Queue] = []
        self._processes: list[mp.Process] = []

        for gpu_id in range(self.data_parallel_size):
            task_queue = ctx.Queue()
            process = ctx.Process(
                target=_dp_worker_main,
                args=(model, gpu_id, self.config_kwargs, task_queue, self._result_queue),
                daemon=False,
            )
            process.start()
            self._task_queues.append(task_queue)
            self._processes.append(process)

        self._collector = threading.Thread(target=self._collect_results, daemon=True)
        self._collector.start()
        atexit.register(self.exit)

    @staticmethod
    def _normalize_messages(
        messages: list[dict] | list[list[dict]],
    ) -> list[list[dict]]:
        if not messages:
            raise ValueError("messages must not be empty")
        if isinstance(messages[0], dict):
            return [messages]  # type: ignore[list-item]
        return messages

    def _collect_results(self) -> None:
        while True:
            req_id, outputs, error = self._result_queue.get()
            with self._pending_lock:
                pending = self._pending.pop(req_id, None)
            if pending is None:
                continue
            pending["outputs"] = outputs
            pending["error"] = error
            pending["event"].set()

    def _submit(
        self,
        messages: list[dict],
        sampling_params: SamplingParams,
    ) -> int:
        if self._closed:
            raise RuntimeError("DPLLMEngine is closed")

        req_id = next(self._request_counter)
        event = threading.Event()
        with self._pending_lock:
            self._pending[req_id] = {
                "event": event,
                "outputs": None,
                "error": None,
            }

        worker_idx = self._round_robin % self.data_parallel_size
        self._round_robin += 1
        self._task_queues[worker_idx].put(
            (req_id, messages, _sampling_params_to_dict(sampling_params))
        )
        return req_id

    def _wait(self, req_id: int) -> list[dict]:
        with self._pending_lock:
            pending = self._pending.get(req_id)
        if pending is None:
            raise KeyError(f"unknown request id: {req_id}")

        pending["event"].wait()
        if pending["error"] is not None:
            raise RuntimeError(pending["error"])
        return pending["outputs"]

    def generate_messages(
        self,
        messages: list[dict] | list[list[dict]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        del use_tqdm
        batch = self._normalize_messages(messages)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(batch)

        req_ids = [
            self._submit(message, sp)
            for message, sp in zip(batch, sampling_params)
        ]
        outputs: list[dict] = []
        for req_id in req_ids:
            outputs.extend(self._wait(req_id))
        return outputs

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        del use_tqdm
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        req_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                req_id = self._submit_text_prompt(prompt, sp)
            else:
                req_id = self._submit_token_prompt(prompt, sp)
            req_ids.append(req_id)

        outputs: list[dict] = []
        for req_id in req_ids:
            outputs.extend(self._wait(req_id))
        return outputs

    def _submit_text_prompt(self, prompt: str, sampling_params: SamplingParams) -> int:
        raise NotImplementedError(
            "text-only generate() is not supported in DP mode; use generate_messages()."
        )

    def _submit_token_prompt(self, prompt: list[int], sampling_params: SamplingParams) -> int:
        raise NotImplementedError(
            "token-id generate() is not supported in DP mode; use generate_messages()."
        )

    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task_queue in self._task_queues:
            task_queue.put(_STOP)
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
