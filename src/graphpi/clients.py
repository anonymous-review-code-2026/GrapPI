from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from typing import Any, Mapping, Sequence

from .config import ModelConfig


class ModelResponseError(ValueError):
    pass


class UsageTracker:
    def __init__(self):
        self._totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "usage_reports": 0, "input_tokens": 0, "output_tokens": 0})

    def record(self, component: str, usage: Any) -> None:
        row = self._totals[component]
        row["calls"] += 1
        if usage is not None:
            row["usage_reports"] += 1
            row["input_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            row["output_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)

    def to_dict(self) -> dict:
        return {name: dict(row) for name, row in sorted(self._totals.items())}


def create_sdk(config: ModelConfig):
    from openai import OpenAI
    key = os.environ.get(config.api_key_env)
    if not key and not config.local:
        raise ValueError(f"Set {config.api_key_env} before making actor API calls")
    return OpenAI(api_key=key or "local", base_url=config.base_url,
                  timeout=config.timeout, max_retries=config.max_retries)


class ChatClient:
    def __init__(self, config: ModelConfig, *, component: str = "actor",
                 tracker: UsageTracker | None = None, sdk=None):
        self.config = config
        self.component = component
        self.tracker = tracker if tracker is not None else UsageTracker()
        self._sdk = sdk

    @property
    def sdk(self):
        if self._sdk is None:
            self._sdk = create_sdk(self.config)
        return self._sdk

    def complete(self, messages: Sequence[Mapping[str, str]], *,
                 schema: dict | None = None, schema_name: str = "response") -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model, "messages": list(messages),
            "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
        }
        if schema is not None:
            if self.config.json_mode == "schema":
                kwargs["response_format"] = {
                    "type": "json_schema", "json_schema": {
                        "name": schema_name, "strict": True, "schema": schema}}
            else:
                kwargs["messages"] = [
                    {"role": "system", "content":
                     ("Return one JSON object using the exact field and row layout in the instructions."
                      if self.config.local else
                      "Return one JSON object conforming to this schema: "
                      + json.dumps(schema, separators=(",", ":")))},
                    *list(messages)]
                kwargs["response_format"] = {"type": "json_object"}
        if self.config.local and self.config.disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.sdk.chat.completions.create(**kwargs)
        self.tracker.record(self.component, response.usage)
        if not response.choices:
            raise ModelResponseError(f"{self.component}: empty completion")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ModelResponseError(f"{self.component}: incomplete completion ({choice.finish_reason})")
        if getattr(choice.message, "refusal", None):
            raise ModelResponseError(f"{self.component}: model refused the request")
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            raise ModelResponseError(f"{self.component}: missing text content")
        return text

    def complete_json(self, messages, *, schema: dict, schema_name: str = "response") -> dict:
        text = self.complete(messages, schema=schema, schema_name=schema_name)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"{self.component}: invalid JSON response") from exc
        if not isinstance(value, dict):
            raise ModelResponseError(f"{self.component}: expected a JSON object")
        return value


class EmbeddingClient:
    def __init__(self, config: ModelConfig, *, tracker: UsageTracker | None = None,
                 batch_size: int = 64, sdk=None):
        if not config.local:
            raise ValueError("Embeddings must use a local service")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.config = config
        self.tracker = tracker if tracker is not None else UsageTracker()
        self.batch_size = batch_size
        self._sdk = sdk
        self._cache: dict[str, tuple[float, ...]] = {}
        self._dimension: int | None = None
        self.stats = {"requested_items": 0, "cache_hits": 0, "unique_inputs": 0,
                      "requests": 0}

    @property
    def sdk(self):
        if self._sdk is None:
            self._sdk = create_sdk(self.config)
        return self._sdk

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be nonempty strings")
        self.stats["requested_items"] += len(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        self.stats["cache_hits"] += sum(t in self._cache for t in texts)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            self.stats["requests"] += 1
            response = self.sdk.embeddings.create(
                model=self.config.model, input=batch, encoding_format="float")
            self.tracker.record("embedding", response.usage)
            indexed = {item.index: item.embedding for item in response.data}
            if len(response.data) != len(batch) or set(indexed) != set(range(len(batch))):
                raise ModelResponseError("Embedding response indices do not match the input batch")
            pending = {}
            dimension = self._dimension
            for index, text in enumerate(batch):
                vector = tuple(float(x) for x in indexed[index])
                if not vector or any(not math.isfinite(x) for x in vector):
                    raise ModelResponseError("Embedding vectors must be finite and nonempty")
                norm = math.hypot(*vector)
                if not math.isfinite(norm) or norm == 0:
                    raise ModelResponseError("Embedding vectors must have a finite nonzero norm")
                if dimension is not None and len(vector) != dimension:
                    raise ModelResponseError("Embedding dimensionality changed")
                dimension = len(vector)
                pending[text] = tuple(x / norm for x in vector)
            self._dimension = dimension
            self._cache.update(pending)
            self.stats["unique_inputs"] += len(pending)
        return [self._cache[t] for t in texts]
