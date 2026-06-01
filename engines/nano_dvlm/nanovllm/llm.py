from nanovllm.engine.dp_engine import DPLLMEngine
from nanovllm.engine.llm_engine import LLMEngine


class LLM:
    """Unified entry point for single-GPU and data-parallel inference."""

    def __init__(self, model: str, **kwargs):
        data_parallel_size = kwargs.pop("data_parallel_size", None)
        if data_parallel_size is None:
            data_parallel_size = kwargs.pop("dp_size", 1)
        data_parallel_size = int(data_parallel_size)

        if data_parallel_size > 1:
            kwargs["data_parallel_size"] = data_parallel_size
            self._engine = DPLLMEngine(model, **kwargs)
        else:
            self._engine = LLMEngine(model, **kwargs)

    @property
    def data_parallel_size(self) -> int:
        return getattr(self._engine, "data_parallel_size", 1)

    @property
    def config(self):
        return getattr(self._engine, "config", None)

    @property
    def tokenizer(self):
        return getattr(self._engine, "tokenizer", None)

    def generate(self, *args, **kwargs):
        return self._engine.generate(*args, **kwargs)

    def generate_messages(self, *args, **kwargs):
        return self._engine.generate_messages(*args, **kwargs)

    def exit(self) -> None:
        self._engine.exit()
