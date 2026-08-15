"""Input contracts and data minimisation for the external LLM boundary."""

import math
import re
from dataclasses import dataclass


class InputValidationError(ValueError):
    """Raised when a request cannot safely enter the LLM boundary."""


_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
_MOBILE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_CASE_ALLOWLIST = {
    "case_id",
    "model_score",
    "model_name",
    "model_version",
    "evidence",
    "federation",
    "data_quality",
    "reviewer_context",
}
_EVIDENCE_ALLOWLIST = {"label", "impact", "value", "confidence"}
_FEDERATION_ALLOWLIST = {"party_count", "alignment_rate", "feature_coverage"}
_IMPACTS = {"up", "down", "neutral"}


def _text(value, field_name, max_length, required=True):
    if value is None:
        if required:
            raise InputValidationError("%s is required" % field_name)
        return ""
    if not isinstance(value, str):
        raise InputValidationError("%s must be a string" % field_name)
    value = value.strip()
    if required and not value:
        raise InputValidationError("%s cannot be empty" % field_name)
    if len(value) > max_length:
        raise InputValidationError("%s exceeds %d characters" % (field_name, max_length))
    return value


def _number(value, field_name, minimum, maximum):
    if isinstance(value, bool):
        raise InputValidationError("%s must be numeric" % field_name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InputValidationError("%s must be numeric" % field_name)
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise InputValidationError("%s must be between %s and %s" % (field_name, minimum, maximum))
    return value


def redact_sensitive_text(value):
    """Remove common direct identifiers from otherwise allowed free-text fields."""

    value = str(value)
    value = _MOBILE_PATTERN.sub("[redacted-mobile]", value)
    value = _ID_CARD_PATTERN.sub("[redacted-id]", value)
    value = _EMAIL_PATTERN.sub("[redacted-email]", value)
    return value


def risk_band(score):
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


@dataclass
class SafeCase:
    """A purpose-limited representation suitable for an external LLM."""

    case_id: str
    model_score: float
    model_name: str
    model_version: str
    evidence: list
    federation: dict
    data_quality: str
    reviewer_context: str
    discarded_fields: list

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise InputValidationError("case must be a JSON object")

        discarded_fields = sorted(str(key) for key in payload if key not in _CASE_ALLOWLIST)
        case_id = _text(payload.get("case_id"), "case_id", 64)
        if not _CASE_ID_PATTERN.match(case_id):
            raise InputValidationError("case_id must be an internal, non-identifying reference")

        score = _number(payload.get("model_score"), "model_score", 0.0, 1.0)
        model_name = _text(payload.get("model_name", "CrediFusion"), "model_name", 80)
        model_version = _text(payload.get("model_version", "unspecified"), "model_version", 80)
        data_quality = _text(payload.get("data_quality", "not_provided"), "data_quality", 80)
        reviewer_context = redact_sensitive_text(
            _text(payload.get("reviewer_context", ""), "reviewer_context", 600, required=False)
        )

        evidence, evidence_discards = _parse_evidence(payload.get("evidence", []))
        federation, federation_discards = _parse_federation(payload.get("federation", {}))
        discarded_fields.extend(evidence_discards)
        discarded_fields.extend(federation_discards)

        return cls(
            case_id=case_id,
            model_score=score,
            model_name=model_name,
            model_version=model_version,
            evidence=evidence,
            federation=federation,
            data_quality=data_quality,
            reviewer_context=reviewer_context,
            discarded_fields=sorted(discarded_fields),
        )

    def prompt_payload(self):
        """Only this allowlisted aggregate is sent to Qwen."""

        return {
            "case_reference": self.case_id,
            "federated_model": {
                "name": self.model_name,
                "version": self.model_version,
                "risk_probability": round(self.model_score, 4),
                "risk_band": risk_band(self.model_score),
            },
            "aggregated_evidence": self.evidence,
            "federation_quality": self.federation,
            "data_quality": self.data_quality,
            "reviewer_context": self.reviewer_context,
        }

    def public_payload(self):
        payload = self.prompt_payload()
        payload["privacy"] = {
            "discarded_field_count": len(self.discarded_fields),
            "discarded_fields": self.discarded_fields,
            "direct_identifiers_sent": False,
        }
        return payload


def _parse_evidence(value):
    if value is None:
        value = []
    if not isinstance(value, list):
        raise InputValidationError("evidence must be an array")
    if len(value) > 12:
        raise InputValidationError("evidence supports at most 12 items")

    cleaned = []
    discarded = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise InputValidationError("evidence[%d] must be an object" % index)
        discarded.extend(
            "evidence[%d].%s" % (index, key) for key in item if key not in _EVIDENCE_ALLOWLIST
        )
        label = redact_sensitive_text(_text(item.get("label"), "evidence.label", 80))
        impact = _text(item.get("impact"), "evidence.impact", 12).lower()
        if impact not in _IMPACTS:
            raise InputValidationError("evidence.impact must be up, down, or neutral")
        evidence_value = redact_sensitive_text(_text(item.get("value"), "evidence.value", 220))
        confidence = _number(item.get("confidence", 0.75), "evidence.confidence", 0.0, 1.0)
        cleaned.append(
            {
                "label": label,
                "impact": impact,
                "value": evidence_value,
                "confidence": round(confidence, 2),
            }
        )
    return cleaned, discarded


def _parse_federation(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise InputValidationError("federation must be an object")
    discarded = ["federation.%s" % key for key in value if key not in _FEDERATION_ALLOWLIST]

    party_count = _number(value.get("party_count", 2), "federation.party_count", 2, 20)
    if party_count != int(party_count):
        raise InputValidationError("federation.party_count must be an integer")
    alignment_rate = _number(value.get("alignment_rate", 0.0), "federation.alignment_rate", 0.0, 1.0)
    feature_coverage = _number(value.get("feature_coverage", 0.0), "federation.feature_coverage", 0.0, 1.0)
    return {
        "party_count": int(party_count),
        "alignment_rate": round(alignment_rate, 3),
        "feature_coverage": round(feature_coverage, 3),
    }, discarded
