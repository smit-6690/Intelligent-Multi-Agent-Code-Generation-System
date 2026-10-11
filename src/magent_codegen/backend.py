from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GenerationBackend(Protocol):
    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        ...


@dataclass
class MockBackend:
    """Deterministic backend used for smoke tests and local development."""

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        del max_tokens
        lowered = prompt.lower()
        if "prime" in lowered:
            return (
                "def solve(n):\n"
                "    if n < 2:\n"
                "        return []\n"
                "    sieve = [True] * (n + 1)\n"
                "    sieve[0] = sieve[1] = False\n"
                "    for value in range(2, int(n ** 0.5) + 1):\n"
                "        if sieve[value]:\n"
                "            sieve[value * value : n + 1 : value] = [False] * (((n - value * value) // value) + 1)\n"
                "    return [value for value, is_prime in enumerate(sieve) if is_prime]\n"
            )
        return "def solve(value):\n    return value\n"


class TransformersBackend:
    """Lazy Hugging Face backend with optional PEFT adapter support."""

    def __init__(self, model_name: str, adapter_path: str | None = None):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the ML dependencies with `pip install -e '.[ml]'`."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install PEFT with `pip install -e '.[ml]'`.") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)
        self._model = model.eval()

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
