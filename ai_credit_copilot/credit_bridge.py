"""Stable adapter for sending a CrediFusion score to the copilot API."""


def build_copilot_request(case_id, model_score, evidence, model_version="unspecified",
                          party_count=2, alignment_rate=0.0, feature_coverage=0.0,
                          data_quality="not_provided"):
    """Build the only payload shape accepted by ``/api/analyze``.

    ``evidence`` must already be aggregated, non-identifying model evidence. Raw
    party features, labels, IDs, names, and free-form customer documents do not
    belong in this adapter.
    """

    return {
        "case": {
            "case_id": case_id,
            "model_score": model_score,
            "model_name": "CrediFusion",
            "model_version": model_version,
            "evidence": evidence,
            "federation": {
                "party_count": party_count,
                "alignment_rate": alignment_rate,
                "feature_coverage": feature_coverage,
            },
            "data_quality": data_quality,
        }
    }
