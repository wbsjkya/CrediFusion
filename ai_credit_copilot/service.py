"""Application service that joins the safe case contract with Qwen."""

from .contracts import InputValidationError, SafeCase
from .prompts import build_analysis_messages, build_chat_messages, parse_structured_output
from .qwen_client import QwenClient


class CreditCopilotService:
    def __init__(self, client=None):
        self.client = client or QwenClient()

    def health(self):
        return self.client.health()

    def analyze(self, payload):
        safe_case = self._case_from_request(payload)
        response = self.client.complete(build_analysis_messages(safe_case), temperature=0.15)
        return {
            "case": safe_case.public_payload(),
            "analysis": {
                "content": response.content,
                "structured": parse_structured_output(response.content),
                "model": response.model,
                "usage": response.usage,
            },
        }

    def chat(self, payload):
        if not isinstance(payload, dict):
            raise InputValidationError("request must be a JSON object")
        safe_case = self._case_from_request(payload)
        question = payload.get("question")
        if not isinstance(question, str) or len(question) > 1600:
            raise InputValidationError("question must be a string up to 1600 characters")
        history = payload.get("history", [])
        if not isinstance(history, list):
            raise InputValidationError("history must be an array")
        response = self.client.complete(build_chat_messages(safe_case, question, history), temperature=0.2)
        return {
            "answer": response.content,
            "model": response.model,
            "usage": response.usage,
            "privacy": safe_case.public_payload()["privacy"],
        }

    @staticmethod
    def _case_from_request(payload):
        if not isinstance(payload, dict):
            raise InputValidationError("request must be a JSON object")
        return SafeCase.from_payload(payload.get("case"))
