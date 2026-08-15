import unittest

from ai_credit_copilot.contracts import (
    InputValidationError,
    SafeCase,
    redact_sensitive_text,
    risk_band,
)


def sample_payload():
    return {
        "case_id": "CASE-2026-0042",
        "model_score": 0.72,
        "model_name": "CrediFusion",
        "model_version": "test-v1",
        "customer_name": "Should never leave the service boundary",
        "federation": {
            "party_count": 2,
            "alignment_rate": 0.96,
            "feature_coverage": 0.91,
            "raw_feature_dump": "not allowed",
        },
        "evidence": [
            {
                "label": "现金流波动",
                "impact": "up",
                "value": "联系号码 13812345678 已被隐藏",
                "confidence": 0.8,
                "id_number": "110101199001011234",
            }
        ],
    }


class SafeCaseTest(unittest.TestCase):
    def test_allowlist_and_redaction_are_applied(self):
        safe_case = SafeCase.from_payload(sample_payload())
        prompt_payload = safe_case.prompt_payload()

        self.assertNotIn("customer_name", prompt_payload)
        self.assertIn("[redacted-mobile]", prompt_payload["aggregated_evidence"][0]["value"])
        self.assertIn("customer_name", safe_case.discarded_fields)
        self.assertIn("evidence[0].id_number", safe_case.discarded_fields)
        self.assertIn("federation.raw_feature_dump", safe_case.discarded_fields)

    def test_direct_identifiers_are_rejected_as_case_references(self):
        for case_id in (
            "customer@example.com",
            "13812345678",
            "CASE-13812345678",
            "CASE-138-1234-5678",
            "110101199001011234",
            "CASE-110101199001011234",
        ):
            with self.subTest(case_id=case_id):
                payload = sample_payload()
                payload["case_id"] = case_id
                with self.assertRaises(InputValidationError):
                    SafeCase.from_payload(payload)

    def test_allowed_metadata_fields_are_redacted(self):
        payload = sample_payload()
        payload["model_name"] = "owner customer@example.com"
        payload["model_version"] = "contact +86 138-1234-5678"
        payload["data_quality"] = "record 110101199001011234"

        prompt_payload = SafeCase.from_payload(payload).prompt_payload()

        self.assertEqual(
            prompt_payload["federated_model"]["name"], "owner [redacted-email]"
        )
        self.assertEqual(
            prompt_payload["federated_model"]["version"], "contact [redacted-mobile]"
        )
        self.assertEqual(prompt_payload["data_quality"], "record [redacted-id]")

    def test_common_mobile_number_formats_are_redacted(self):
        for mobile in (
            "13812345678",
            "138-1234-5678",
            "138 1234 5678",
            "+86 13812345678",
            "0086-138-1234-5678",
        ):
            with self.subTest(mobile=mobile):
                self.assertEqual(
                    redact_sensitive_text("contact " + mobile),
                    "contact [redacted-mobile]",
                )

    def test_risk_band_boundaries(self):
        self.assertEqual(risk_band(0.39), "low")
        self.assertEqual(risk_band(0.40), "medium")
        self.assertEqual(risk_band(0.70), "high")
