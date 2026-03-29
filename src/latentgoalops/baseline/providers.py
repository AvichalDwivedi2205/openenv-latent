"""OpenAI-compatible provider adapter with retry and checkpoint-friendly errors."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI


GRADIENT_BASE_URL = "https://inference.do-ai.run/v1"
MODEL_PRICING = {
    "openai-gpt-oss-20b": {"input": 0.05 / 1_000_000, "output": 0.05 / 1_000_000},
    "openai-gpt-oss-120b": {"input": 0.10 / 1_000_000, "output": 0.10 / 1_000_000},
    "alibaba-qwen3-32b": {"input": 0.25 / 1_000_000, "output": 0.55 / 1_000_000},
    "deepseek-r1-distill-llama-70b": {"input": 0.99 / 1_000_000, "output": 0.99 / 1_000_000},
}


class ProviderExhaustedError(RuntimeError):
    """Raised when billing or account limits stop the run."""


class ProviderCredentialError(RuntimeError):
    """Raised when the API token is invalid or revoked."""


@dataclass(slots=True)
class ProviderResponse:
    """Structured provider response."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict | None = None


class GradientChatProvider:
    """Small chat-completions wrapper for OpenAI-compatible endpoints."""

    def __init__(self, model_name: str, max_retries: int = 3, temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.temperature = temperature
        base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or GRADIENT_BASE_URL
        base_url_lower = base_url.lower()
        if "do-ai.run" in base_url_lower or "digitalocean" in base_url_lower:
            token = (
                os.getenv("DIGITALOCEAN_API_TOKEN")
                or os.getenv("HF_TOKEN")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("MODEL_ACCESS_KEY")
            )
        else:
            token = (
                os.getenv("HF_TOKEN")
                or os.getenv("DIGITALOCEAN_API_TOKEN")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("MODEL_ACCESS_KEY")
            )
        if not token:
            raise ValueError(
                "Set HF_TOKEN (or DIGITALOCEAN_API_TOKEN / OPENAI_API_KEY / MODEL_ACCESS_KEY) "
                "before running model baselines."
            )
        self.client = OpenAI(base_url=base_url, api_key=token)

    def complete(self, messages: list[dict], max_tokens: int = 700) -> ProviderResponse:
        """Run a chat completion with retry handling."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=self.temperature,
                    reasoning_effort="low",
                )
                usage = response.usage
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                pricing = MODEL_PRICING.get(self.model_name, {"input": 0.0, "output": 0.0})
                cost_usd = input_tokens * pricing["input"] + output_tokens * pricing["output"]
                message = response.choices[0].message
                content = message.content or "{}"
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return ProviderResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    raw=response.model_dump(mode="json"),
                )
            except APIStatusError as exc:
                last_error = exc
                message = str(exc).lower()
                if exc.status_code in (401, 403):
                    raise ProviderCredentialError("Gradient token rejected the request.") from exc
                if exc.status_code == 429 or 500 <= exc.status_code < 600:
                    if "quota" in message or "billing" in message or "credit" in message:
                        raise ProviderExhaustedError("Provider reported quota or billing exhaustion.") from exc
                    time.sleep(min(2**attempt, 8))
                    continue
                raise
            except APIConnectionError as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
                continue
        raise RuntimeError(f"Provider request failed after retries: {last_error}")
