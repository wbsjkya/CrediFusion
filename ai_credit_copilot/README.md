# CrediFusion AI Credit Copilot

This module adds a local, auditable review layer around a CrediFusion risk score. It does not alter the federated training loop. The local model remains responsible for the score; Qwen is limited to explaining an already-produced, de-identified summary and answering the reviewer's follow-up questions.

## What is included

- Qwen-compatible chat interface through DashScope's OpenAI-compatible endpoint.
- A local browser workbench for risk-signal explanation, reviewer questions, and original-output traceability.
- Strict request allowlist: only a case reference, score, aggregate evidence, federation quality, and data-quality notes cross the LLM boundary.
- Common mobile numbers, Chinese ID-card numbers, and email addresses are redacted in allowed free-text fields. Unknown fields are dropped and returned in the audit summary.
- A small `credit_bridge.py` adapter that converts a CrediFusion result to the API request contract.

## Run locally

From the `CrediFusion` directory, configure a DashScope key in the current PowerShell session:

```powershell
$env:QWEN_API_KEY = "your_dashscope_api_key"
$env:QWEN_MODEL = "qwen3.5"
python -m ai_credit_copilot --port 8787
```

Open `http://127.0.0.1:8787` in a browser. The server uses only the Python standard library, so it adds no package dependency to the existing PyTorch/FATE research environment.

`DASHSCOPE_API_KEY` is also supported. `QWEN_BASE_URL` defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`; set it only when using an approved compatible gateway. The default model string is `qwen3.5`. If the provider exposes a more specific model identifier for the account, set `QWEN_MODEL` to that exact identifier instead.

## Connect a model output

After the federated inference pipeline produces a probability and aggregate feature evidence, build the request without including raw party features or customer identifiers:

```python
from ai_credit_copilot.credit_bridge import build_copilot_request

payload = build_copilot_request(
    case_id="CASE-2026-0042",
    model_score=0.72,
    model_version="credit-prod-2026.08",
    party_count=2,
    alignment_rate=0.96,
    feature_coverage=0.91,
    evidence=[
        {
            "label": "Recent repayment volatility",
            "impact": "up",
            "value": "Aggregated delinquency signals require human verification.",
            "confidence": 0.86,
        }
    ],
)
```

Send `payload` to `POST /api/analyze`, or use the same `case` object with `POST /api/chat` and a `question` field. The API contract is intentionally narrow. Never add names, phone numbers, account numbers, addresses, source documents, raw party features, labels, embeddings, or feature dumps.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Shows the selected Qwen model and whether a key is configured. |
| `GET` | `/api/demo-case` | Returns a synthetic, de-identified sample case. |
| `POST` | `/api/analyze` | Produces structured review assistance from an allowed case summary. |
| `POST` | `/api/chat` | Answers a reviewer question in the same safe case context. |

## Safety boundary

The copilot prompt prevents approval, rejection, credit-limit, interest-rate, or pricing decisions. Its output is a review aid, not an automated decision. It cannot replace model validation, fairness testing, human review, data-governance approval, or regulatory review.

## Verify

```powershell
python -m unittest discover -s ai_credit_copilot/tests -v
```
