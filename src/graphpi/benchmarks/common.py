from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        if isinstance(raw, dict):
            for name in ("tasks", "examples", "runs", "results", "cases"):
                if name in raw:
                    raw = raw[name]
                    break
    if not isinstance(raw, list) or not raw or any(not isinstance(row, dict) for row in raw):
        raise ValueError("Expected a non-empty JSON array or JSONL of objects")
    return raw


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path: str | Path, values: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def require_texts(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(require_text(item, field) for item in value)


def require_score(value: Any, field: str) -> int:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{field} must contain exactly one score")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
        raise ValueError(f"{field} must be an integer in [0, 10]")
    return value


def require_unique_ids(rows: Sequence[Any]) -> None:
    ids = [row.id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task IDs")


def select_ids(rows: Sequence[Any], identifiers: Sequence[str] | None = None,
               limit: int | None = None) -> list[Any]:
    selected = list(rows)
    if identifiers:
        wanted = set(identifiers)
        missing = wanted - {row.id for row in selected}
        if missing:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
        selected = [row for row in selected if row.id in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Cannot average missing or non-finite scores")
    return sum(values) / len(values)
