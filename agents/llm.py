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
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_BASE_URL_ENV = "CREDITPULSE_LLM_BASE_URL"
DEFAULT_MODEL_ENV = "CREDITPULSE_LLM_MODEL"
DEFAULT_KEY_ENV = "CREDITPULSE_LLM_API_KEY"

JUDGE_MODEL_ENV = "CREDITPULSE_JUDGE_MODEL"


@dataclass
class Completion:
    """A model reply: free text, native tool calls, or both.

    Modern instruct models emit tool calls through the API's own mechanism
    rather than as JSON in prose, and several do so regardless of prompting.
    Carrying both shapes lets the loop accept whichever the model produces.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: The provider's assistant message, verbatim.
    #:
    #: Replayed rather than reconstructed. Providers attach fields to tool
    #: calls that must be echoed back unchanged -- Gemini 3.x rejects a
    #: conversation whose function calls have lost their thought_signature --
    #: and rebuilding the message from name and arguments silently drops them.
    raw_message: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _dispatch_complete_call(
    client: Any, messages: Sequence[dict[str, Any]], tools: list[dict] | None = None, **kwargs: Any
) -> Completion:
    """Use a client's native tool calling if it has any, else adapt its text.

    Keeps the wrappers (cache, rate limiter) agnostic: a ScriptedClient that
    only speaks text still works unchanged through the same path.
    """
    if hasattr(client, "complete_call"):
        return client.complete_call(messages, tools=tools, **kwargs)
    return Completion(content=client.complete(messages, **kwargs))


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
    max_tokens: int = 1600
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

    def complete_call(
        self, messages: Sequence[dict[str, Any]], tools: list[dict] | None = None, **kwargs: Any
    ) -> Completion:
        """A completion that may carry native tool calls.

        Falls back to the text protocol when the provider rejects the model's
        own tool call. gpt-oss-20b intermittently leaks harmony control tokens
        into the function name -- "get_metric<|channel|>analysis" -- and the
        provider 400s the whole request. Retrying identically at temperature 0
        would reproduce it, so the retry drops the tool declarations and lets
        the model answer as text instead.
        """
        try:
            return self._complete_call(messages, tools, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if tools and "tool_use_failed" in str(exc):
                return self._complete_call(messages, None, **kwargs)
            raise

    def _complete_call(
        self, messages: Sequence[dict[str, Any]], tools: list[dict] | None = None, **kwargs: Any
    ) -> Completion:
        if not self.model:
            raise RuntimeError(f"no model configured; set {DEFAULT_MODEL_ENV}")
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key or "not-needed", base_url=self.base_url or None)
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = "auto"
        response = client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **extra,
        )
        message = response.choices[0].message
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": call.id, "name": call.function.name, "arguments": args})
        try:
            raw = message.model_dump(exclude_none=True)
        except AttributeError:
            raw = None
        return Completion(content=message.content or "", tool_calls=calls, raw_message=raw)


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

    def _key(self, messages: Sequence[dict[str, str]], tools: Any = None) -> str:
        """Hash the model, the conversation, and the tools that were offered.

        The tool list belongs in the key. Without it, toggling a tool on or off
        while leaving the prompt untouched is invisible to the cache, so an
        arm run *without* the baseline tool would silently replay completions
        generated *with* it -- the two arms would differ only in the flag, not
        in the answers. That is the exact contamination this cache exists to
        make impossible, so the schema names are part of the identity.
        """
        names = sorted(t.get("function", {}).get("name", "") for t in (tools or []))
        payload = json.dumps(
            {
                "model": getattr(self.inner, "name", ""),
                "messages": list(messages),
                "tools": names,
            },
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


    def complete_call(self, messages, tools=None, **kwargs):
        """Cache native tool calls too, keyed by the same conversation hash."""
        key = self._key(messages, tools) + ("_t" if tools else "")
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            data = json.loads(path.read_text(encoding="utf8"))
            return Completion(
                content=data["content"],
                tool_calls=data["tool_calls"],
                raw_message=data.get("raw_message"),
            )
        result = _dispatch_complete_call(self.inner, messages, tools, **kwargs)
        path.write_text(
            json.dumps(
                {
                    "content": result.content,
                    "tool_calls": result.tool_calls,
                    "raw_message": result.raw_message,
                }
            ),
            encoding="utf8",
        )
        self.misses += 1
        return result

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


#: Read once at import so a key never has to be typed into a shell or a chat.
ENV_FILE = Path(".env")


def load_env_file(path: Path | str = ENV_FILE) -> list[str]:
    """Load ``KEY=value`` pairs from a gitignored ``.env``.

    Exists because the alternative is worse. A key pasted into a terminal or a
    message is exposed the moment it is written, and provider secret-scanning
    revokes exposed keys within minutes -- which is exactly how the first Groq
    key here died mid-session. A file that git ignores and nothing echoes is
    the only handling that does not leak.

    Returns the names it set, never the values.
    """
    path = Path(path)
    if not path.exists():
        return []
    loaded: list[str] = []
    for line in path.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


class BudgetExhausted(RuntimeError):
    """Raised when the run hits its own spending cap, not the provider's."""


class InfrastructureError(RuntimeError):
    """The endpoint is unusable: bad key, no credits, forbidden.

    Kept distinct from any agent-quality signal. A 402 swallowed as a per-case
    failure once produced an L3 report reading "94.8% protocol failure", which
    blamed the model for an unpaid account -- a wrong conclusion stated
    confidently, which is the exact failure mode this project exists to catch.
    These abort the run instead.
    """


@dataclass
class RateLimitedClient:
    """Retries on rate limits and stops before a budget is exhausted.

    Free tiers limit tokens per minute and per day. Two distinct problems:

    * **Transient** (per-minute): the right response is to wait and retry.
      Providers send ``retry-after``; otherwise exponential backoff.
    * **Terminal** (daily cap): retrying cannot help. The run should stop
      cleanly while whatever it completed is already in the response cache,
      so tomorrow resumes instead of restarting.

    ``max_calls`` is a self-imposed ceiling, deliberately separate from the
    provider's. Hitting our own limit stops the run in a known state; hitting
    theirs means discovering it mid-request with a partial conversation.
    """

    inner: Any
    max_calls: int = 0
    max_retries: int = 5
    base_delay: float = 2.0
    #: Proactive tokens-per-minute ceiling. 0 disables pacing.
    #:
    #: Reactive backoff alone wastes throughput: it fires until the provider
    #: returns 429, then sleeps exponentially, which overshoots. Measured on a
    #: 16K-TPM model, that idled at ~8.7K TPM -- barely half the allowance.
    #: Spacing calls to sit just under the ceiling avoids the 429 entirely.
    tokens_per_minute: int = 0
    name: str = ""
    calls: int = 0
    retries: int = 0
    waited_seconds: float = 0.0
    paced_seconds: float = 0.0
    _window: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = f"ratelimited:{getattr(self.inner, 'name', 'unknown')}"

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if status == 429:
            return True
        text = str(exc).lower()
        return "429" in text or "rate limit" in text or "too many requests" in text

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """Network blips and provider 5xx: worth retrying, unlike a bad request.

        A three-hour unattended run will meet at least one dropped connection.
        Treating that as fatal threw away 40 completed cases the first time.
        """
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if isinstance(status, int) and 500 <= status < 600:
            return True
        name = type(exc).__name__.lower()
        if any(k in name for k in ("connection", "timeout", "apierror", "internalserver")):
            return True
        text = str(exc).lower()
        return any(
            k in text
            for k in ("connection error", "timed out", "timeout", "temporarily unavailable",
                      "bad gateway", "service unavailable", "502", "503", "504")
        )

    @staticmethod
    def _is_daily_cap(exc: Exception) -> bool:
        """A daily or credit limit, which retrying cannot clear.

        Deliberately narrow. Matching the bare word "quota" was wrong: Gemini
        phrases its *per-minute* limit as "You exceeded your current quota"
        with quotaId GenerateRequestsPerMinutePerProjectPerModel, so a
        transient throttle was being treated as terminal and killed the run.
        A per-minute signal wins over any generic quota wording.
        """
        text = str(exc).lower().replace(" ", "")
        if "perminute" in text or "requestsperminute" in text:
            return False
        return any(
            phrase in text
            for phrase in ("perday", "daily", "insufficient_quota", "insufficientcredit", "credits")
        )

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        import re

        match = re.search(r"retry in ([0-9.]+)(ms|s)", str(exc), re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value / 1000.0 if match.group(2).lower() == "ms" else value
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            raw = headers.get(key)
            if not raw:
                continue
            try:
                return float(str(raw).rstrip("s"))
            except ValueError:
                continue
        return None

    @staticmethod
    def _estimate_tokens(messages: Sequence[dict[str, Any]], tools: list | None) -> int:
        """Rough token count for pacing. ~4 characters per token.

        Deliberately approximate: pacing only needs to be close enough to stay
        under the ceiling, and an exact count would mean tokenising every
        request.
        """
        size = sum(len(str(m.get("content") or "")) for m in messages)
        size += sum(len(json.dumps(m.get("tool_calls") or [])) for m in messages)
        if tools:
            size += len(json.dumps(tools))
        return int(size / 4) + 200  # + headroom for the reply

    def _pace(self, estimated: int) -> None:
        """Sleep just long enough to stay under the per-minute ceiling."""
        if not self.tokens_per_minute:
            return
        now = time.monotonic()
        self._window = [(t, n) for t, n in self._window if now - t < 60.0]
        used = sum(n for _, n in self._window)
        if used + estimated > self.tokens_per_minute and self._window:
            wait = 60.0 - (now - self._window[0][0]) + 0.25
            if wait > 0:
                time.sleep(wait)
                self.paced_seconds += wait
                now = time.monotonic()
                self._window = [(t, n) for t, n in self._window if now - t < 60.0]
        self._window.append((time.monotonic(), estimated))

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        self._pace(self._estimate_tokens(messages, None))
        return self._with_retries(lambda: self.inner.complete(messages, **kwargs))

    def _with_retries(self, action):
        if self.max_calls and self.calls >= self.max_calls:
            raise BudgetExhausted(
                f"self-imposed budget of {self.max_calls} calls reached; "
                "completed work is cached, so a re-run resumes rather than repeats"
            )
        delay = self.base_delay
        for attempt in range(self.max_retries + 1):
            try:
                result = action()
                self.calls += 1
                return result
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                text = str(exc).lower()
                if status in (401, 402, 403) or "insufficient credit" in text:
                    raise InfrastructureError(
                        f"endpoint unusable ({status}): {str(exc)[:200]}"
                    ) from exc
                if not (self._is_rate_limit(exc) or self._is_transient(exc)):
                    raise
                if self._is_rate_limit(exc) and self._is_daily_cap(exc):
                    raise BudgetExhausted(
                        f"provider daily/credit limit reached: {str(exc)[:160]}. "
                        "Completed work is cached; re-run when the quota resets."
                    ) from exc
                if attempt >= self.max_retries:
                    raise
                wait = self._retry_after(exc) or delay
                wait = min(wait, 60.0)
                self.retries += 1
                self.waited_seconds += wait
                time.sleep(wait)
                delay *= 2
        raise RuntimeError("unreachable")


    def complete_call(self, messages, tools=None, **kwargs):
        self._pace(self._estimate_tokens(messages, tools))
        return self._with_retries(
            lambda: _dispatch_complete_call(self.inner, messages, tools, **kwargs)
        )

    def stats(self) -> dict[str, float]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "waited_seconds": round(self.waited_seconds, 1),
            "paced_seconds": round(self.paced_seconds, 1),
        }


#: Default agent model. A ~20B mixture-of-experts: only a few billion
#: parameters are active per token, so it costs and runs like a small model
#: while reasoning like a mid-tier one -- the right trade for a backtest that
#: is throughput-bound rather than capability-bound.
#:
#: The measured floor is below it: 0.5B could not hold the tool protocol at
#: all, and 1.5B fabricated a metric it had never fetched. Whether 20B is
#: actually needed over 8B is an open question the harness is built to answer.
DEFAULT_AGENT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"

#: Providers exposing an OpenAI-compatible endpoint. Model IDs churn -- check
#: the provider's current list if one 404s.
KNOWN_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "vllm": "http://localhost:8000/v1",
}


def default_client(model: str = "", base_url: str = "") -> LLMClient:
    """The configured agent model, or a clear error explaining what is missing."""
    model = model or os.environ.get(DEFAULT_MODEL_ENV, "")
    key = os.environ.get(DEFAULT_KEY_ENV, "")
    base = base_url or os.environ.get(DEFAULT_BASE_URL_ENV, "")
    if model and (key or base):
        return OpenAICompatibleClient(model=model, base_url=base, api_key=key)
    raise RuntimeError(
        "no agent model configured. Set:\n"
        f"  {DEFAULT_BASE_URL_ENV}   e.g. {KNOWN_ENDPOINTS['groq']}\n"
        f"  {DEFAULT_MODEL_ENV}      e.g. {DEFAULT_AGENT_MODEL}\n"
        f"  {DEFAULT_KEY_ENV}        your provider key\n"
        "The L0-L2 suite, the deterministic baselines and the rule-based "
        "control all run without one."
    )


#: Reasoning models (the gpt-oss family among them) emit reasoning tokens
#: before any content, and those count against ``max_tokens``. A tight budget
#: returns ``finish_reason=length`` with empty content, which looks exactly
#: like a broken endpoint. Measured: gpt-oss-20b used all 16 tokens on
#: reasoning and returned nothing; at 400 it answered correctly in 72.
PREFLIGHT_MAX_TOKENS = 400


def preflight(client: LLMClient) -> tuple[bool, str]:
    """One cheap round-trip to confirm the endpoint answers.

    Worth two seconds: a bad model ID or key would otherwise surface partway
    through a 400-case backtest, after real tokens had been spent.
    """
    try:
        reply = client.complete(
            [{"role": "user", "content": 'Reply with exactly: {"ok": true}'}],
            max_tokens=PREFLIGHT_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if not (reply or "").strip():
        return False, (
            "endpoint returned empty content. If this is a reasoning model, "
            "reasoning tokens consumed the whole budget -- raise max_tokens"
        )
    return True, reply.strip()[:80]


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
