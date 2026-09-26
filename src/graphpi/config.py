from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import ipaddress
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


def validate_endpoint(url: str, *, local: bool) -> None:
    if not isinstance(url, str):
        raise ValueError("Endpoint URL must be a string")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid endpoint port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Endpoint port must be between 1 and 65535")
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("Endpoints must be HTTP(S) /v1 URLs without credentials or query parameters")
    if local:
        host = parsed.hostname
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            raise ValueError("Graph and embedding services must use a loopback endpoint")


@dataclass(frozen=True)
class ModelConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 2
    local: bool = False
    json_mode: str = "schema"
    disable_thinking: bool = False

    def __post_init__(self):
        validate_endpoint(self.base_url, local=self.local)
        if (not isinstance(self.model, str) or not self.model.strip()
                or not isinstance(self.api_key_env, str) or not self.api_key_env.strip()
                or "=" in self.api_key_env or "\0" in self.api_key_env):
            raise ValueError("Model and API key environment variable names must be valid strings")
        if type(self.local) is not bool or type(self.disable_thinking) is not bool:
            raise ValueError("Local service flags must be booleans")
        if (type(self.temperature) not in {int, float}
                or type(self.timeout) not in {int, float}
                or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("Invalid model sampling or timeout configuration")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be an integer between zero and five")
        if self.json_mode not in {"schema", "object"}:
            raise ValueError("json_mode must be schema or object")
        if self.disable_thinking and not self.local:
            raise ValueError("Local model options cannot be sent to the actor endpoint")


def _builder():
    return ModelConfig(model="graphpi-builder", base_url="http://127.0.0.1:8001/v1",
                       api_key_env="GRAPHPI_LOCAL_API_KEY", local=True,
                       max_tokens=8192, disable_thinking=True)


def _linker():
    return ModelConfig(model="graphpi-linker", base_url="http://127.0.0.1:8002/v1",
                       api_key_env="GRAPHPI_LOCAL_API_KEY", local=True,
                       max_tokens=512, disable_thinking=True)


def _embedding():
    return ModelConfig(model="BAAI/bge-m3", base_url="http://127.0.0.1:8003/v1",
                       api_key_env="GRAPHPI_LOCAL_API_KEY", local=True)


@dataclass(frozen=True)
class AlgorithmConfig:
    neighbors: int = 3
    pair_limit: int = 96
    retrieval_top_k: int = 24
    evidence_budget: int = 4
    budget_scope: str = "episode"
    rounds: int = 6
    max_nodes_per_source: int = 128
    embedding_batch_size: int = 64
    builder_batch_chars: int = 12000
    builder_batch_units: int = 64

    def __post_init__(self):
        for name in ("neighbors", "pair_limit", "retrieval_top_k",
                     "max_nodes_per_source", "embedding_batch_size",
                     "builder_batch_chars", "builder_batch_units"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("evidence_budget", "rounds"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.budget_scope not in {"episode", "receive"}:
            raise ValueError("budget_scope must be episode or receive")


@dataclass(frozen=True)
class GraphPIConfig:
    actor: ModelConfig = field(default_factory=ModelConfig)
    builder: ModelConfig = field(default_factory=_builder)
    linker: ModelConfig = field(default_factory=_linker)
    embedding: ModelConfig = field(default_factory=_embedding)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    def __post_init__(self):
        if self.actor.local:
            raise ValueError("Actor configuration must use the standard actor API contract")
        if any(not getattr(self, name).local for name in ("builder", "linker", "embedding")):
            raise ValueError("All GraphPI auxiliary components must be local")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> GraphPIConfig:
    if path is None:
        return GraphPIConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(raw) - {f.name for f in fields(GraphPIConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    defaults = GraphPIConfig().to_dict()
    for name, updates in raw.items():
        if not isinstance(updates, dict):
            raise ValueError(f"Configuration section {name} must be an object")
        defaults[name].update(updates)
    return GraphPIConfig(
        **{name: ModelConfig(**defaults[name])
           for name in ("actor", "builder", "linker", "embedding")},
        algorithm=AlgorithmConfig(**defaults["algorithm"]),
    )
