from __future__ import annotations

import unittest

from novelty_app.agents.orchestrator_langgraph import (
    AuditReport,
    ClaimSupportStatus,
    node_audit,
    route_after_audit,
)


class _FakeStructuredInvoker:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": messages, "config": config})
        return self.response


class _FakeLLM:
    def __init__(self, response):
        self.structured = _FakeStructuredInvoker(response)
        self.structured_schema = None
        self.structured_method = None

    def with_structured_output(self, schema, method=None):
        self.structured_schema = schema
        self.structured_method = method
        return self.structured


class OrchestratorAuditTests(unittest.TestCase):
    def test_audit_report_accepts_explicit_support_status_enum(self) -> None:
        report = AuditReport.model_validate(
            {
                "supported_claim_fraction": 1.0,
                "needs_patch": False,
                "claims": [
                    {"claim": "fully grounded", "support_status": "supported"},
                    {"claim": "only partly grounded", "support_status": "partial"},
                    {"claim": "not grounded", "support_status": "unsupported"},
                ],
            }
        )

        self.assertEqual(report.claims[0].support_status, ClaimSupportStatus.supported)
        self.assertEqual(report.claims[1].support_status, ClaimSupportStatus.partial)
        self.assertEqual(report.claims[2].support_status, ClaimSupportStatus.unsupported)
        self.assertAlmostEqual(report.supported_claim_fraction, 0.5)
        self.assertTrue(report.needs_patch)

    def test_audit_report_accepts_legacy_supported_values(self) -> None:
        report = AuditReport.model_validate(
            {
                "supported_claim_fraction": 0.0,
                "needs_patch": False,
                "claims": [
                    {"claim": "legacy partial", "supported": "partially"},
                    {"claim": "legacy true", "supported": True},
                    {"claim": "legacy false", "supported": False},
                ],
            }
        )

        self.assertEqual(report.claims[0].support_status, ClaimSupportStatus.partial)
        self.assertEqual(report.claims[1].support_status, ClaimSupportStatus.supported)
        self.assertEqual(report.claims[2].support_status, ClaimSupportStatus.unsupported)
        self.assertAlmostEqual(report.supported_claim_fraction, 0.5)

    def test_route_after_audit_patches_when_partial_support_exists(self) -> None:
        audit = AuditReport.model_validate(
            {
                "supported_claim_fraction": 1.0,
                "needs_patch": False,
                "claims": [{"claim": "indirectly supported", "support_status": "partial"}],
            }
        ).model_dump(mode="json")

        self.assertEqual(route_after_audit({"audit": audit, "iter": 0, "max_iters": 2}), "patch")
        self.assertEqual(route_after_audit({"audit": audit, "iter": 2, "max_iters": 2}), "ideate")

    def test_node_audit_serializes_support_status_as_json_strings(self) -> None:
        llm = _FakeLLM(
            AuditReport.model_validate(
                {
                    "supported_claim_fraction": 1.0,
                    "needs_patch": False,
                    "claims": [{"claim": "indirectly supported", "support_status": "partial"}],
                }
            )
        )
        state = {
            "target_type": "gap",
            "gap_id": "gap_1",
            "snapshot_id": "snapshot_1",
            "explanation": {"bridge_seeds": []},
            "evidence": [{"paper_id": "p1", "title": "Paper 1", "abstract": "Text"}],
            "discovery_cue": {"text": "focus"},
        }

        out = node_audit(state, llm)

        self.assertEqual(llm.structured_method, "function_calling")
        self.assertEqual(out["audit"]["claims"][0]["support_status"], "partial")
        self.assertNotIn("supported", out["audit"]["claims"][0])
        self.assertTrue(out["audit"]["needs_patch"])


if __name__ == "__main__":
    unittest.main()
