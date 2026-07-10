"""Tests for model adapters."""

import pytest
from unittest.mock import Mock, patch

from secdrift.models import get_model, list_models, ModelAdapter, ModelResponse
from secdrift.models.base import ModelAdapter, ModelResponse
from secdrift.models.providers import BedrockAdapter, OpenAICompatibleAdapter


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_model_response_success(self):
        response = ModelResponse(
            text="def hello(): pass",
            model="test-model",
            provider="test",
            latency_ms=100.0,
            success=True,
        )
        assert response.success
        assert response.text == "def hello(): pass"
        assert response.error is None

    def test_model_response_failure(self):
        response = ModelResponse(
            text="",
            model="test-model",
            provider="test",
            latency_ms=50.0,
            success=False,
            error="Connection timeout",
        )
        assert not response.success
        assert response.error == "Connection timeout"


class TestModelAdapter:
    """Tests for ModelAdapter base class."""

    def test_extract_code_markdown(self):
        adapter = BedrockAdapter(model_id="test")
        text = """Here's the code:
```python
def hello():
    print("world")
```
That's it!"""
        code = adapter.extract_code(text)
        assert "def hello():" in code
        assert "print" in code

    def test_extract_code_no_markdown(self):
        adapter = BedrockAdapter(model_id="test")
        text = "def hello():\n    pass"
        code = adapter.extract_code(text)
        assert code == text


class TestListModels:
    """Tests for model listing."""

    def test_list_models_returns_list(self):
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_models_contains_expected(self):
        models = list_models()
        # Should have Claude models
        assert any("claude" in m for m in models)


class TestGetModel:
    """Tests for model retrieval."""

    def test_get_model_valid(self):
        # This will work if config exists
        model = get_model("claude-sonnet-4-6")
        assert isinstance(model, ModelAdapter)
        assert model.model_id is not None

    def test_get_model_invalid(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent-model-xyz")
