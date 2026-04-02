from __future__ import annotations

import hashlib
import inspect
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from types import MethodType
from typing import Any, Dict, Iterator, List, Optional

try:
    from langfuse import Langfuse, propagate_attributes
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore
    propagate_attributes = None  # type: ignore

try:
    from langfuse.langchain import CallbackHandler as LangfuseLangchainCallbackHandler
except Exception:  # pragma: no cover
    try:
        from langfuse.callback import CallbackHandler as LangfuseLangchainCallbackHandler  # type: ignore
    except Exception:  # pragma: no cover
        LangfuseLangchainCallbackHandler = None  # type: ignore


_TRUTHY = {"1", "true", "yes", "on"}
_LANGFUSE_TRACE_ATTRIBUTE_KEYS = frozenset(
    {"langfuse_session_id", "langfuse_user_id", "langfuse_tags"}
)


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


def _sanitize_propagated_callback_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metadata is None or not isinstance(metadata, dict):
        return metadata
    preserved: Dict[str, Any] = {}
    to_stringify: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key in _LANGFUSE_TRACE_ATTRIBUTE_KEYS:
            preserved[key] = value
            continue
        to_stringify[key] = value
    sanitized = _stringify_trace_metadata(to_stringify)
    sanitized.update(preserved)
    return sanitized


def _patch_langfuse_callback_for_propagation(callback: Any) -> Any:
    parser = getattr(callback, "_parse_langfuse_trace_attributes", None)
    if not callable(parser) or getattr(callback, "_novelty_trace_attrs_sanitized", False):
        return callback

    def _patched_parse_langfuse_trace_attributes(
        self,
        *,
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        return parser(
            metadata=_sanitize_propagated_callback_metadata(metadata),
            tags=tags,
        )

    try:
        callback._parse_langfuse_trace_attributes = MethodType(  # type: ignore[attr-defined]
            _patched_parse_langfuse_trace_attributes,
            callback,
        )
        callback._novelty_trace_attrs_sanitized = True  # type: ignore[attr-defined]
    except Exception:
        return callback
    return callback


def _langfuse_base_url() -> Optional[str]:
    return os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or None


def _langfuse_constructor_kwargs() -> Dict[str, Any]:
    tracing_enabled = _env_enabled("LANGFUSE_TRACING_ENABLED", default=True)
    kwargs: Dict[str, Any] = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "base_url": _langfuse_base_url(),
        "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or None,
        # Support both old and new SDK parameter names; unsupported fields are filtered at call time.
        "tracing_enabled": tracing_enabled,
        "enabled": tracing_enabled,
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
        return Langfuse(**_filter_callable_kwargs(Langfuse, _langfuse_constructor_kwargs()))
    except Exception:
        return None


def _filter_callable_kwargs(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return kwargs
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value not in (None, "", [], {})
    }


def langfuse_enabled() -> bool:
    return get_langfuse_client() is not None


def langfuse_status() -> Dict[str, Any]:
    base_url = _langfuse_base_url()
    tracing_enabled = _env_enabled("LANGFUSE_TRACING_ENABLED", default=True)
    has_public_key = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    has_secret_key = bool(os.getenv("LANGFUSE_SECRET_KEY"))
    status: Dict[str, Any] = {
        "enabled": False,
        "sdk_available": Langfuse is not None,
        "callback_available": LangfuseLangchainCallbackHandler is not None,
        "tracing_enabled_env": tracing_enabled,
        "has_public_key": has_public_key,
        "has_secret_key": has_secret_key,
    }
    if base_url:
        status["base_url"] = base_url
    if Langfuse is None:
        status["reason"] = "langfuse_sdk_unavailable"
        return status
    if not tracing_enabled:
        status["reason"] = "tracing_disabled_by_env"
        return status
    if not has_public_key or not has_secret_key:
        status["reason"] = "missing_langfuse_keys"
        return status
    client = get_langfuse_client()
    if client is None:
        status["reason"] = "langfuse_client_init_failed"
        return status
    status["enabled"] = True
    status["reason"] = "ok"
    return status


@dataclass
class _NullObservation:
    observation_id: Optional[str] = None

    def update(self, **_kwargs: Any) -> None:
        return None

    def score(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def get_langfuse_langchain_callback(
    *,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    trace_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    if LangfuseLangchainCallbackHandler is None:
        return None
    if get_langfuse_client() is None:
        return None

    kwargs: Dict[str, Any] = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "host": _langfuse_base_url(),
        "session_id": session_id,
        "trace_name": trace_name,
        "tags": list(tags or []),
        "metadata": _stringify_trace_metadata(metadata),
        "enabled": _env_enabled("LANGFUSE_TRACING_ENABLED", default=True),
    }
    filtered_kwargs = _filter_callable_kwargs(LangfuseLangchainCallbackHandler, kwargs)
    try:
        callback = LangfuseLangchainCallbackHandler(**filtered_kwargs)
    except Exception:
        return None
    return _patch_langfuse_callback_for_propagation(callback)


def langchain_config_with_observability(
    config: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    trace_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    callback = get_langfuse_langchain_callback(
        session_id=session_id,
        tags=tags,
        trace_name=trace_name,
        metadata=metadata,
    )
    if callback is None:
        return dict(config) if config else None

    merged = dict(config or {})
    callbacks = list(merged.get("callbacks") or [])
    callbacks.append(callback)
    merged["callbacks"] = callbacks
    return merged


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
    try:
        trace_id = client.get_current_trace_id()
    except Exception:
        return {}
    if not trace_id:
        return {}
    try:
        observation_id = client.get_current_observation_id()
    except Exception:
        observation_id = None
    trace_url: Optional[str] = None
    try:
        trace_url = client.get_trace_url(trace_id=trace_id)
    except Exception:
        trace_url = None
    ref: Dict[str, Any] = {
        "provider": "langfuse",
        "trace_id": trace_id,
        "observation_id": observation_id,
        "url": trace_url,
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
