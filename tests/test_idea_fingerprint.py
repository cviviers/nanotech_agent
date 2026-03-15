from __future__ import annotations

import unittest

from novelty_app.evaluation.idea_fingerprint import fingerprint_hypothesis


class IdeaFingerprintTests(unittest.TestCase):
    def test_fingerprint_hypothesis_extracts_domain_terms(self) -> None:
        fp = fingerprint_hypothesis(
            {
                "title": "Folate liposome siRNA bridge for breast cancer",
                "text": "Use a liposome with folate targeting to deliver siRNA in breast cancer models.",
                "mechanistic_rationale": "Improved targeting and gene silencing.",
            }
        )
        self.assertIn("liposome", fp["material"])
        self.assertIn("folate", fp["targeting"])
        self.assertIn("breast", fp["disease"])
        self.assertIn("sirna", fp["payload"])


if __name__ == "__main__":
    unittest.main()
