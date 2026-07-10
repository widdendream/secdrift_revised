"""Model provider implementations.

Supported providers:
- bedrock: AWS Bedrock Converse API
- openai: OpenAI Chat Completions API
- anthropic: Anthropic Messages API
- openai_compatible: Any OpenAI-compatible API (DeepSeek, Together, Ollama, vLLM, etc.)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from secdrift.models.base import ModelAdapter, ModelResponse

logger = logging.getLogger(__name__)


class BedrockAdapter(ModelAdapter):
    """AWS Bedrock adapter using Converse API.

    Uses standard AWS credential chain (environment variables, ~/.aws/credentials, IAM roles).
    """

    provider = "bedrock"

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        profile: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, **kwargs)
        self.region = region
        self.profile = profile
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            session_kwargs = {}
            if self.profile:
                session_kwargs["profile_name"] = self.profile

            session = boto3.Session(**session_kwargs)
            config = Config(
                region_name=self.region,
                retries={"max_attempts": self.max_retries, "mode": "adaptive"}
            )
            self._client = session.client("bedrock-runtime", config=config)
        return self._client

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate using Bedrock Converse API."""
        start_time = time.perf_counter()

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                }
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract text from response
            output_text = ""
            if response.get("output", {}).get("message", {}).get("content"):
                content = response["output"]["message"]["content"]
                output_text = content[0].get("text", "") if content else ""

            # Get token usage
            usage = response.get("usage", {})

            return ModelResponse(
                text=output_text,
                model=self.model_id,
                provider=self.provider,
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                latency_ms=latency_ms,
                success=True,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Bedrock error: {e}")
            return ModelResponse(
                text="",
                model=self.model_id,
                provider=self.provider,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def is_available(self) -> bool:
        """Check if Bedrock credentials are available."""
        try:
            import boto3
            session_kwargs = {}
            if self.profile:
                session_kwargs["profile_name"] = self.profile
            session = boto3.Session(**session_kwargs)
            return session.get_credentials() is not None
        except Exception:
            return False


class OpenAIAdapter(ModelAdapter):
    """OpenAI Chat Completions API adapter."""

    provider = "openai"

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        **kwargs
    ):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate using OpenAI Chat API."""
        start_time = time.perf_counter()

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return ModelResponse(
                text=response.choices[0].message.content or "",
                model=self.model_id,
                provider=self.provider,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency_ms,
                success=True,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"OpenAI error: {e}")
            return ModelResponse(
                text="",
                model=self.model_id,
                provider=self.provider,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def is_available(self) -> bool:
        return self.api_key is not None


class AnthropicAdapter(ModelAdapter):
    """Anthropic Messages API adapter."""

    provider = "anthropic"

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate using Anthropic Messages API."""
        start_time = time.perf_counter()

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            output_text = response.content[0].text if response.content else ""

            return ModelResponse(
                text=output_text,
                model=self.model_id,
                provider=self.provider,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                success=True,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Anthropic error: {e}")
            return ModelResponse(
                text="",
                model=self.model_id,
                provider=self.provider,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def is_available(self) -> bool:
        return self.api_key is not None


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for OpenAI-compatible APIs.

    Works with: DeepSeek, Together AI, Fireworks, vLLM, Ollama, etc.
    """

    provider = "openai_compatible"

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        provider_name: str = "openai_compatible",
        **kwargs
    ):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider_name  # Allow custom provider name for logging
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key or "not-needed",
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate using OpenAI-compatible API."""
        start_time = time.perf_counter()

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return ModelResponse(
                text=response.choices[0].message.content or "",
                model=self.model_id,
                provider=self.provider,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency_ms,
                success=True,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"{self.provider} error: {e}")
            return ModelResponse(
                text="",
                model=self.model_id,
                provider=self.provider,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def is_available(self) -> bool:
        # For local providers, try to connect
        if "localhost" in self.base_url:
            try:
                import requests
                requests.get(self.base_url.replace("/v1", ""), timeout=2)
                return True
            except Exception:
                return False
        return self.api_key is not None


# Provider type mapping
PROVIDER_CLASSES = {
    "bedrock": BedrockAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "openai_compatible": OpenAICompatibleAdapter,
}
