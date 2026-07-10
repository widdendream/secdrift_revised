"""Base model adapter interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ModelResponse:
    """Response from a model generation call."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    All model providers (Bedrock, OpenAI, etc.) implement this interface.
    """

    provider: str = "base"

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 90,
        max_retries: int = 3,
        **kwargs
    ):
        """Initialize the adapter.

        Args:
            model_id: Provider-specific model identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: The user prompt
            **kwargs: Additional model-specific parameters

        Returns:
            ModelResponse with generated text and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model/provider is available."""
        pass

    def extract_code(self, text: str) -> str:
        """Extract code from model response.

        Handles markdown code blocks and plain code.
        """
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        return text.strip()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id!r})"
