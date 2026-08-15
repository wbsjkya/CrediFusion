# CrediFusion

[English](README.md) | [中文](README-CN.md)

> Privacy-preserving self-supervised vertical federated learning and human-in-the-loop credit intelligence.

CrediFusion is a research and engineering prototype for learning useful representations across organizations while keeping raw party data local. It combines self-supervised vertical federated learning experiments with a local, privacy-aware review layer for already-produced model signals.

## Why the name CrediFusion?

`Credi` blends **credit** and **credibility**: it represents credit-risk assessment as well as the project's focus on trustworthy, reviewable model assistance. `Fusion` reflects the privacy-preserving fusion of representations and risk signals learned by multiple parties through vertical federated learning, together with the collaboration between model output and human review.

Together, **CrediFusion** means a trusted credit-intelligence system that brings cross-party knowledge together without moving raw data out of the institutions that own it.

## What is included

- **Federated training code** in `colla/`, `local/`, and `finetune/` for self-supervised pretraining, vertical collaboration, and downstream fine-tuning experiments.
- **Credit review copilot** in `ai_credit_copilot/`, including a standard-library HTTP service, browser workbench, Qwen-compatible client, input allowlist, redaction, and audit-oriented response handling.
- **Tests and safe examples** for the review service. The included case is synthetic and de-identified.

The public copy intentionally contains source code, tests, a configuration template, synthetic example data, and README files only.

## Repository layout

```text
CrediFusion/
|-- colla/                     # Collaborative FATE training examples
|-- local/                     # Local SSL pretraining and classification code
|-- finetune/                  # Downstream federated fine-tuning example
|-- ai_credit_copilot/         # Review service and browser workbench
|   |-- credit_bridge.py       # Safe model-output adapter
|   |-- sample_case.json       # Synthetic de-identified demo case
|   `-- tests/                 # Unit tests for the review boundary
`-- README.md
```

## Quick start: review workbench

The review service uses only the Python standard library and can run independently of the PyTorch/FATE research environment.

From the repository root:

```powershell
$env:QWEN_API_KEY = "your_dashscope_api_key"
$env:QWEN_MODEL = "qwen3.5"
python -m ai_credit_copilot --port 8787
```

Open `http://127.0.0.1:8787` in a browser. `DASHSCOPE_API_KEY` is also supported. `QWEN_BASE_URL` defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`; set it only when using an approved compatible gateway. Never commit API keys to Git.

To connect an aggregate risk output from the federated model:

```python
from ai_credit_copilot.credit_bridge import build_copilot_request

payload = build_copilot_request(
    case_id="CASE-2026-0042",
    model_score=0.72,
    model_version="credit-demo-2026.08",
    party_count=2,
    alignment_rate=0.96,
    feature_coverage=0.91,
    evidence=[
        {
            "label": "Recent repayment volatility",
            "impact": "up",
            "value": "Aggregated repayment signals require human verification.",
            "confidence": 0.86,
        }
    ],
)
```

Send `payload` to `POST /api/analyze`. Use `POST /api/chat` for a reviewer follow-up in the same constrained case context. Only aggregated, non-identifying evidence belongs in the payload; raw party features, labels, embeddings, source documents, and direct identifiers must stay outside the LLM boundary.

## Run the research examples

The research scripts require a compatible FATE, PyTorch, scikit-learn, and dataset environment. Dataset paths and deployment settings are intentionally left to the user; no experiment data is bundled in this public copy.

```powershell
cd colla
python stage1.py --parties guest:9999 host:10000 --log_level INFO
```

The `local/` directory contains local pretraining and downstream classification scripts. `finetune/` contains a downstream collaborative fine-tuning example.

## Verify

```powershell
python -m unittest discover -s ai_credit_copilot/tests -v
```

## Safety and scope

CrediFusion is a research prototype, not an automated credit-decision system. The copilot may explain model signals and prepare human-review questions, but it must not approve, reject, price, or determine credit limits. Any real deployment requires approved data, privacy and security review, model validation, fairness evaluation, human oversight, and applicable compliance review.
