"""Agent-facing backend utilities and orchestrators."""

try:  # pragma: no cover - optional import dependency (requests)
    from agents.backend_client import BackendClient
except Exception:  # pragma: no cover
    BackendClient = None  # type: ignore

__all__ = ["BackendClient"]
