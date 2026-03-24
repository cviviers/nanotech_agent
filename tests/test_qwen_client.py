from __future__ import annotations

import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

class QwenClientTests(unittest.TestCase):
    @staticmethod
    def _load_module():
        fake_requests = types.SimpleNamespace()

        class FakeRequestException(Exception):
            pass

        class FakeConnectionError(FakeRequestException):
            pass

        fake_requests.RequestException = FakeRequestException
        fake_requests.ConnectionError = FakeConnectionError
        fake_requests.post = Mock()

        with patch.dict(sys.modules, {"requests": fake_requests}):
            module = importlib.import_module("novelty_app.evaluation.qwen_client")
            module = importlib.reload(module)
        return module, fake_requests

    def test_embed_prints_http_error_locally(self) -> None:
        module, _fake_requests = self._load_module()
        client = module.QwenClient(base_url="http://fake")
        response = Mock()
        response.ok = False
        response.status_code = 503
        response.json.return_value = {"detail": "embed failed"}
        response.text = '{"detail":"embed failed"}'

        stderr = io.StringIO()
        with patch.object(module.requests, "post", return_value=response):
            with redirect_stderr(stderr):
                with self.assertRaises(RuntimeError):
                    client.embed(["hello"])

        output = stderr.getvalue()
        self.assertIn("POST /embed failed with HTTP 503", output)
        self.assertIn("embed failed", output)

    def test_embed_prints_request_exception_locally(self) -> None:
        module, fake_requests = self._load_module()
        client = module.QwenClient(base_url="http://fake")

        stderr = io.StringIO()
        with patch.object(
            module.requests,
            "post",
            side_effect=fake_requests.ConnectionError("connection dropped"),
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(fake_requests.ConnectionError):
                    client.embed(["hello"])

        self.assertIn("POST /embed request error: connection dropped", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
