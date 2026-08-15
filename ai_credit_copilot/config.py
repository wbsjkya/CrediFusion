"""Runtime configuration for the Qwen-compatible client."""

import os
from dataclasses import dataclass


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.5"


def _positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True)
class QwenSettings:
    """Environment-backed settings kept outside source control."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: int
    max_tokens: int

    @classmethod
    def from_env(cls):
        return cls(
            api_key=(os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip(),
            base_url=(os.getenv("QWEN_BASE_URL") or DEFAULT_QWEN_BASE_URL).strip(),
            model=(os.getenv("QWEN_MODEL") or DEFAULT_QWEN_MODEL).strip(),
            timeout_seconds=_positive_int(os.getenv("QWEN_TIMEOUT_SECONDS"), 45),
            max_tokens=_positive_int(os.getenv("QWEN_MAX_TOKENS"), 1400),
        )

    @property
    def chat_completions_url(self):
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return base_url + "/chat/completions"

    @property
    def is_configured(self):
        return bool(self.api_key and self.base_url and self.model)
