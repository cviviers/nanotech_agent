from __future__ import annotations

import hashlib
import os
import unittest
from unittest.mock import patch

from novelty_app.agents.observability import (
    current_trace_ref,
    deterministic_trace_id,
    get_langfuse_langchain_callback,
    langfuse_status,
    langchain_config_with_observability,
    observe_current,
)


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


class _StrictCallbackHandler:
    def __init__(self, *, session_id=None, tags=None, enabled=None):
        self.session_id = session_id
        self.tags = tags
        self.enabled = enabled


class _ParsingCallbackHandler:
    def __init__(self, *, session_id=None, tags=None, enabled=None, metadata=None):
        self.session_id = session_id
        self.tags = tags
        self.enabled = enabled
        self.metadata = metadata

    def _parse_langfuse_trace_attributes(self, *, metadata=None, tags=None):
        attributes = {}
        metadata = dict(metadata or {})
        if isinstance(metadata.get("langfuse_session_id"), str):
            attributes["session_id"] = metadata["langfuse_session_id"]
        merged_tags = list(tags or [])
        if isinstance(metadata.get("langfuse_tags"), list):
            merged_tags.extend(str(tag) for tag in metadata["langfuse_tags"])
        if merged_tags:
            attributes["tags"] = merged_tags
        attributes["metadata"] = {
            key: value
            for key, value in metadata.items()
            if key not in {"langfuse_session_id", "langfuse_user_id", "langfuse_tags"}
        }
        return attributes


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

    def test_langchain_config_with_observability_appends_callback(self) -> None:
        with patch("novelty_app.agents.observability.get_langfuse_client", return_value=object()), patch(
            "novelty_app.agents.observability.LangfuseLangchainCallbackHandler",
            _StrictCallbackHandler,
        ):
            config = langchain_config_with_observability(
                {"callbacks": ["existing"]},
                session_id="session_1",
                tags=["judge"],
                trace_name="ignored",
                metadata={"kind": "unit"},
            )

        self.assertIsNotNone(config)
        self.assertEqual(config["callbacks"][0], "existing")
        self.assertIsInstance(config["callbacks"][1], _StrictCallbackHandler)
        self.assertEqual(config["callbacks"][1].session_id, "session_1")
        self.assertEqual(config["callbacks"][1].tags, ["judge"])

    def test_langchain_callback_sanitizes_langgraph_metadata_for_propagation(self) -> None:
        with patch("novelty_app.agents.observability.get_langfuse_client", return_value=object()), patch(
            "novelty_app.agents.observability.LangfuseLangchainCallbackHandler",
            _ParsingCallbackHandler,
        ):
            callback = get_langfuse_langchain_callback(session_id="session_1", tags=["judge"])

        self.assertIsNotNone(callback)
        parsed = callback._parse_langfuse_trace_attributes(
            metadata={
                "langgraph_step": 3,
                "langgraph_node": "ideate",
                "langgraph_triggers": ["branch:to:ideate"],
                "langgraph_path": ("root", "ideate"),
                "langfuse_session_id": "session_from_metadata",
                "langfuse_tags": ["from_metadata"],
            },
            tags=["judge"],
        )
        self.assertEqual(parsed["session_id"], "session_from_metadata")
        self.assertEqual(parsed["tags"], ["judge", "from_metadata"])
        self.assertEqual(
            parsed["metadata"],
            {
                "langgraph_step": "3",
                "langgraph_node": "ideate",
                "langgraph_triggers": "['branch:to:ideate']",
                "langgraph_path": "('root', 'ideate')",
            },
        )

    def test_langfuse_status_reports_missing_keys(self) -> None:
        with patch("novelty_app.agents.observability.Langfuse", object), patch.dict("os.environ", {}, clear=False):
            status = langfuse_status()

        self.assertFalse(status["enabled"])
        self.assertEqual(status["reason"], "missing_langfuse_keys")

    def test_langfuse_status_reports_enabled_when_client_available(self) -> None:
        with patch("novelty_app.agents.observability.Langfuse", object), patch(
            "novelty_app.agents.observability.LangfuseLangchainCallbackHandler",
            object,
        ), patch.dict(
            "os.environ",
            {
                "LANGFUSE_PUBLIC_KEY": "pk_test",
                "LANGFUSE_SECRET_KEY": "sk_test",
                "LANGFUSE_BASE_URL": "http://localhost:3000",
                "LANGFUSE_TRACING_ENABLED": "true",
            },
            clear=False,
        ), patch("novelty_app.agents.observability.get_langfuse_client", return_value=object()):
            status = langfuse_status()

        self.assertTrue(status["enabled"])
        self.assertEqual(status["reason"], "ok")
        self.assertEqual(status["base_url"], "http://localhost:3000")

    def test_get_langfuse_client_filters_constructor_kwargs_for_sdk_version(self) -> None:
        from novelty_app.agents import observability

        class _TracingEnabledOnlyLangfuse:
            def __init__(
                self,
                *,
                public_key=None,
                secret_key=None,
                base_url=None,
                tracing_enabled=None,
                environment=None,
                sample_rate=None,
            ):
                self.kwargs = {
                    "public_key": public_key,
                    "secret_key": secret_key,
                    "base_url": base_url,
                    "tracing_enabled": tracing_enabled,
                    "environment": environment,
                    "sample_rate": sample_rate,
                }

        observability.get_langfuse_client.cache_clear()
        with patch.object(observability, "Langfuse", _TracingEnabledOnlyLangfuse), patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk_test",
                "LANGFUSE_SECRET_KEY": "sk_test",
                "LANGFUSE_BASE_URL": "http://localhost:3000",
                "LANGFUSE_TRACING_ENABLED": "true",
            },
            clear=False,
        ):
            client = observability.get_langfuse_client()
        observability.get_langfuse_client.cache_clear()

        self.assertIsInstance(client, _TracingEnabledOnlyLangfuse)
        self.assertTrue(client.kwargs["tracing_enabled"])
        self.assertEqual(client.kwargs["base_url"], "http://localhost:3000")


if __name__ == "__main__":
    unittest.main()
