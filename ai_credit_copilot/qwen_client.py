"""Small dependency-free Qwen client using DashScope's OpenAI-compatible API."""

import json
from dataclasses import dataclass
from urllib import error, request

from .config import QwenSettings


class QwenError(RuntimeError):
    pass


class QwenConfigurationError(QwenError):
    pass


class QwenRequestError(QwenError):
    pass


@dataclass
class QwenResponse:
    content: str
    model: str
    usage: dict


class QwenClient:
    """Calls a configurable Qwen-compatible chat-completions endpoint."""

    def __init__(self, settings=None):
        self.settings = settings or QwenSettings.from_env()

    def health(self):
        return {
            "provider": "DashScope OpenAI-compatible API",
            "model": self.settings.model,
            "configured": self.settings.is_configured,
            "endpoint": self.settings.chat_completions_url,
        }

    def complete(self, messages, temperature=0.2, max_tokens=None):
        if not self.settings.is_configured:
            raise QwenConfigurationError(
                "QWEN_API_KEY (or DASHSCOPE_API_KEY), QWEN_BASE_URL, and QWEN_MODEL must be configured"
            )
        if not isinstance(messages, list) or not messages:
            raise QwenRequestError("messages must be a non-empty array")

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": max(0.0, min(float(temperature), 1.0)),
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            self.settings.chat_completions_url,
            data=body,
            headers={
                "Authorization": "Bearer " + self.settings.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "credifusion-ai-copilot/0.1",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.settings.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raise QwenRequestError("Qwen API returned HTTP %s" % exc.code)
        except error.URLError as exc:
            raise QwenRequestError("Qwen API connection failed: %s" % exc.reason)
        except OSError as exc:
            raise QwenRequestError("Qwen API request failed: %s" % exc)

        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise QwenRequestError("Qwen API returned an unexpected response shape")

        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("text")
            )
        if not isinstance(content, str) or not content.strip():
            raise QwenRequestError("Qwen API returned an empty response")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return QwenResponse(content=content.strip(), model=data.get("model", self.settings.model), usage=usage)
