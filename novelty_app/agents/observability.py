from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

try:
    from langfuse import Langfuse, propagate_attributes
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore
    propagate_attributes = None  # type: ignore


_TRUTHY = {"1", "true", "yes", "on"}


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def _stringify_trace_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in (metadata or {}).items():
        text = str(value).strip()
        if not text:
            continue
        key_text = str(key).strip()
        if not key_text:
            continue
        out[key_text[:200]] = text[:200]
    return out


def _langfuse_constructor_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "base_url": os.getenv("LANGFUSE_BASE_URL") or None,
        "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or None,
        "enabled": _env_enabled("LANGFUSE_TRACING_ENABLED", default=True),
    }
    sample_rate = os.getenv("LANGFUSE_SAMPLE_RATE")
    if sample_rate not in {None, ""}:
        try:
            kwargs["sample_rate"] = float(sample_rate)
        except ValueError:
            pass
    return kwargs


@lru_cache(maxsize=1)
def get_langfuse_client() -> Any:
    if Langfuse is None:
        return None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        return Langfuse(**_langfuse_constructor_kwargs())
    except Exception:
        return None


def langfuse_enabled() -> bool:
    return get_langfuse_client() is not None


@dataclass
class _NullObservation:
    observation_id: Optional[str] = None

    def update(self, **_kwargs: Any) -> None:
        return None

    def score(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def deterministic_trace_id(seed: str) -> str:
    client = get_langfuse_client()
    if client is not None:
        try:
            return str(client.create_trace_id(seed=seed))
        except Exception:
            pass
    return hashlib.sha256(seed.encode("utf-8")).digest()[:16].hex()


@contextmanager
def observe_current(
    *,
    name: str,
    as_type: str = "span",
    input_payload: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    model: Optional[str] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    client = get_langfuse_client()
    if client is None:
        yield _NullObservation()
        return

    kwargs: Dict[str, Any] = {
        "name": name,
        "as_type": as_type,
        "input": input_payload,
        "metadata": metadata,
    }
    if trace_id:
        kwargs["trace_context"] = {"trace_id": trace_id}
    if model:
        kwargs["model"] = model
    if model_parameters:
        kwargs["model_parameters"] = model_parameters

    with client.start_as_current_observation(**kwargs) as observation:
        try:
            yield observation
        except Exception as exc:
            try:
                observation.update(level="ERROR", status_message=str(exc))
            except Exception:
                pass
            raise


@contextmanager
def trace_attributes(
    *,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    trace_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    if propagate_attributes is None or get_langfuse_client() is None:
        with nullcontext():
            yield
        return

    with propagate_attributes(
        session_id=session_id,
        tags=list(tags or []),
        trace_name=trace_name,
        metadata=_stringify_trace_metadata(metadata),
    ):
        yield


def current_trace_ref(
    *,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = get_langfuse_client()
    if client is None:
        return {}
    trace_id = client.get_current_trace_id()
    if not trace_id:
        return {}
    ref: Dict[str, Any] = {
        "provider": "langfuse",
        "trace_id": trace_id,
        "observation_id": client.get_current_observation_id(),
        "url": client.get_trace_url(trace_id=trace_id),
        "session_id": session_id,
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
    }
    return {key: value for key, value in ref.items() if value not in (None, "", [], {})}


def create_trace_score(
    *,
    trace_ref: Optional[Dict[str, Any]],
    name: str,
    value: Any,
    data_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    client = get_langfuse_client()
    if client is None:
        return
    trace_id = str((trace_ref or {}).get("trace_id") or "")
    if not trace_id:
        return
    try:
        client.create_score(
            name=name,
            value=value,
            trace_id=trace_id,
            session_id=(trace_ref or {}).get("session_id"),
            data_type=data_type,
            metadata=metadata,
        )
    except Exception:
        return


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        return
