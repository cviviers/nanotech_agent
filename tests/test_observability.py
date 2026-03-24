from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

from novelty_app.agents.observability import current_trace_ref, deterministic_trace_id, observe_current


class _FakeObservation:
    def __init__(self, client, trace_id: str, observation_id: str):
        self._client = client
        self.trace_id = trace_id
        self.observation_id = observation_id
        self.updated = []

    def __enter__(self):
        self._client.stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._client.stack.pop()
        return False

    def update(self, **kwargs):
        self.updated.append(kwargs)


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.stack = []
        self.counter = 0

    def create_trace_id(self, *, seed=None):
        return hashlib.sha256(str(seed or "seed").encode("utf-8")).digest()[:16].hex()

    def start_as_current_observation(self, *, trace_context=None, **_kwargs):
        trace_id = (trace_context or {}).get("trace_id") or self.create_trace_id(seed=f"trace-{self.counter + 1}")
        self.counter += 1
        return _FakeObservation(self, trace_id, f"{self.counter:016x}")

    def get_current_trace_id(self):
        return self.stack[-1].trace_id if self.stack else None

    def get_current_observation_id(self):
        return self.stack[-1].observation_id if self.stack else None

    def get_trace_url(self, *, trace_id=None):
        resolved = trace_id or self.get_current_trace_id()
        return f"https://langfuse.local/project/test/traces/{resolved}" if resolved else None


class ObservabilityTests(unittest.TestCase):
    def test_deterministic_trace_id_is_stable_without_client(self) -> None:
        with patch("novelty_app.agents.observability.get_langfuse_client", return_value=None):
            self.assertEqual(deterministic_trace_id("abc"), deterministic_trace_id("abc"))
            self.assertEqual(len(deterministic_trace_id("abc")), 32)

    def test_observe_current_is_noop_without_client(self) -> None:
        with patch("novelty_app.agents.observability.get_langfuse_client", return_value=None):
            with observe_current(name="noop") as observation:
                observation.update(output={"ok": True})
                self.assertEqual(current_trace_ref(), {})

    def test_current_trace_ref_uses_active_fake_client(self) -> None:
        fake_client = _FakeLangfuseClient()
        with patch("novelty_app.agents.observability.get_langfuse_client", return_value=fake_client):
            with observe_current(name="workflow", as_type="agent", trace_id=fake_client.create_trace_id(seed="workflow")):
                ref = current_trace_ref(session_id="session_1", tags=["test"], metadata={"kind": "unit"})
        self.assertEqual(ref["provider"], "langfuse")
        self.assertEqual(ref["session_id"], "session_1")
        self.assertEqual(ref["tags"], ["test"])
        self.assertEqual(ref["metadata"]["kind"], "unit")
        self.assertIn("/traces/", ref["url"])


if __name__ == "__main__":
    unittest.main()
