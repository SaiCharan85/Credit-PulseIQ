"""Model clients behind a narrow protocol.

Three implementations:

:class:`OpenAICompatibleClient`
    Talks to any OpenAI-compatible endpoint, which is what vLLM serves
    (SPEC 11). Point ``CREDITPULSE_LLM_BASE_URL`` at it.
:class:`ScriptedClient`
    Returns a fixed sequence of replies. This is how the ReAct loop is tested
    without a model: the script can exercise multi-step tool selection,
    recovery from a tool error, and termination at *insufficient evidence*,
    deterministically and offline.
:class:`EchoJudgeClient`
    Placeholder for the separate judge (hard rule 7); never the generating model.

The loop depends only on :class:`LLMClient`, so swapping a real endpoint in is a
configuration change, not a code change.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_BASE_URL_ENV = "CREDITPULSE_LLM_BASE_URL"
DEFAULT_MODEL_ENV = "CREDITPULSE_LLM_MODEL"
DEFAULT_KEY_ENV = "CREDITPULSE_LLM_API_KEY"

JUDGE_MODEL_ENV = "CREDITPULSE_JUDGE_MODEL"


class LLMClient(Protocol):
    """Minimal chat interface. Deliberately small: the loop, not the client,
    owns tool dispatch and control flow."""

    name: str

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass
class ScriptedClient:
    """Replays a fixed list of assistant messages.

    Used by the L3 loop tests. A scripted model is not a mock of the loop --
    the loop really parses these replies, really dispatches the tools, and
    really decides when to stop; only the token generation is fixed.
    """

    script: list[str] = field(default_factory=list)
    name: str = "scripted"
    calls: list[list[dict[str, str]]] = field(default_factory=list)
    fallback: str = (
        '{"action": "finish", "signal": "insufficient_evidence", '
        '"confidence": 0.2, "rationale": "script exhausted"}'
    )

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            return self.fallback
        return self.script.pop(0)


@dataclass
class OpenAICompatibleClient:
    """Any OpenAI-compatible endpoint, including a local vLLM server."""

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    name: str = "openai-compatible"

    def __post_init__(self) -> None:
        self.model = self.model or os.environ.get(DEFAULT_MODEL_ENV, "")
        self.base_url = self.base_url or os.environ.get(DEFAULT_BASE_URL_ENV, "")
        self.api_key = self.api_key or os.environ.get(DEFAULT_KEY_ENV, "not-needed")
        self.name = f"openai-compatible:{self.model or 'unset'}"

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        if not self.model:
            raise RuntimeError(
                f"no model configured; set {DEFAULT_MODEL_ENV} (and "
                f"{DEFAULT_BASE_URL_ENV} for a local vLLM server)"
            )
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key or "not-needed", base_url=self.base_url or None
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content or ""


@dataclass
class TransformersClient:
    """A local Hugging Face instruct model, for machines without a GPU.

    A fallback, not the target. SPEC 11 calls for a strong instruct model on
    vLLM; this exists so the loop can be exercised end to end on CPU. Small
    models follow the JSON tool protocol unreliably, and the loop's abstention
    path will absorb that -- which makes "the agent abstained" ambiguous
    between honest uncertainty and the model being too weak to comply. Treat
    results from this client as a wiring check, not as evidence.
    """

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 220
    temperature: float = 0.0
    name: str = ""
    _pipe: Any = None

    def __post_init__(self) -> None:
        self.name = f"transformers:{self.model_id}"

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            model.eval()
            self._pipe = (tokenizer, model)
        return self._pipe

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        import torch

        tokenizer, model = self._load()
        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


@dataclass
class CachingClient:
    """Caches completions on disk, keyed by model and exact conversation.

    Hosted providers are not bit-reproducible even at temperature 0, so an
    uncached backtest gives different numbers on every run and a regression
    cannot be distinguished from provider drift. For a project whose central
    claim is reproducibility that is not acceptable.

    It also makes iteration nearly free: re-running a backtest after changing
    only the grading code replays the cached completions instead of paying for
    them again.
    """

    inner: Any
    cache_dir: Path = Path("data/cache/llm")
    name: str = ""
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.name = f"cached:{getattr(self.inner, 'name', 'unknown')}"
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages: Sequence[dict[str, str]]) -> str:
        payload = json.dumps(
            {"model": getattr(self.inner, "name", ""), "messages": list(messages)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf8")).hexdigest()[:32]

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        path = self.cache_dir / f"{self._key(messages)}.txt"
        if path.exists():
            self.hits += 1
            return path.read_text(encoding="utf8")
        reply = self.inner.complete(messages, **kwargs)
        path.write_text(reply, encoding="utf8")
        self.misses += 1
        return reply

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


def default_client() -> LLMClient:
    """The configured agent model, or a clear error explaining what is missing."""
    if os.environ.get(DEFAULT_MODEL_ENV):
        return OpenAICompatibleClient()
    raise RuntimeError(
        f"no agent model configured. Set {DEFAULT_MODEL_ENV} and optionally "
        f"{DEFAULT_BASE_URL_ENV} (e.g. http://localhost:8000/v1 for vLLM). "
        "The L0-L2 suite and the deterministic baselines run without one."
    )


def judge_client() -> LLMClient:
    """The judge must be a different model from the generator (hard rule 7)."""
    model = os.environ.get(JUDGE_MODEL_ENV)
    if not model:
        raise RuntimeError(f"no judge model configured; set {JUDGE_MODEL_ENV}")
    if model == os.environ.get(DEFAULT_MODEL_ENV):
        raise RuntimeError(
            "the judge model must differ from the agent model: grading an output "
            "with the model that produced it invites self-preference bias"
        )
    return OpenAICompatibleClient(model=model)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a reply.

    Instruct models wrap JSON in prose or fences even when told not to. Parsing
    leniently here and validating strictly downstream keeps formatting slips
    from being scored as reasoning failures.
    """
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                cleaned = candidate
                break
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None
