"""Gemini adapter and selector tests with fully mocked SDK clients."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.decisions.schemas import DecisionPlan
from app.embeddings.provider import (
    EMBEDDING_DIMENSION,
    EmbeddingProviderError,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from app.llm.provider import GeminiLLMProvider, LLMProviderError, OpenAILLMProvider


class FakeGeminiModels:
    def __init__(self, *, embeddings=None, text="A grounded response.", error=None):
        self.embeddings = embeddings
        self.text = text
        self.error = error
        self.embed_calls = []
        self.generate_calls = []

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(embeddings=self.embeddings)

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


class FakeGeminiClient:
    def __init__(self, models):
        self.models = models


def _reset_provider_cache(module):
    if hasattr(module, "_embedding_provider_instance"):
        module._embedding_provider_instance = None
        module._embedding_provider_resolved = False
    if hasattr(module, "_llm_provider_instance"):
        module._llm_provider_instance = None
        module._llm_provider_resolved = False


def test_gemini_embedding_returns_one_1536d_vector_per_input():
    from google.genai import types

    models = FakeGeminiModels(embeddings=[
        SimpleNamespace(values=[0.1] * EMBEDDING_DIMENSION),
        SimpleNamespace(values=[0.2] * EMBEDDING_DIMENSION),
    ])
    provider = GeminiEmbeddingProvider("fake-key", client=FakeGeminiClient(models))

    vectors = provider.embed(["first passage", "second passage"])

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    call = models.embed_calls[0]
    assert call["model"] == "gemini-embedding-2"
    assert call["config"].output_dimensionality == EMBEDDING_DIMENSION
    assert len(call["contents"]) == 2
    assert all(isinstance(content, types.Content) for content in call["contents"])
    assert [content.parts[0].text for content in call["contents"]] == ["first passage", "second passage"]


def test_gemini_embedding_empty_input_skips_api_call():
    models = FakeGeminiModels()
    assert GeminiEmbeddingProvider("fake-key", client=FakeGeminiClient(models)).embed([]) == []
    assert models.embed_calls == []


def test_gemini_embedding_failure_is_normalized_for_existing_best_effort_path():
    models = FakeGeminiModels(error=RuntimeError("bad Gemini key"))
    provider = GeminiEmbeddingProvider("invalid-key", client=FakeGeminiClient(models))
    with pytest.raises(EmbeddingProviderError, match="Gemini embedding request failed"):
        provider.embed(["text"])


def test_gemini_llm_plans_and_generates_grounded_answers():
    plan_json = (
        '{"requires_decisions":false,"workflow":null,"combination":null,"decisions":[]}'
    )
    models = FakeGeminiModels(text=plan_json)
    provider = GeminiLLMProvider("fake-key", model="test-model", client=FakeGeminiClient(models))

    plan = provider.plan_decisions("question", "grounded context")
    assert isinstance(plan, DecisionPlan)
    plan_call = models.generate_calls[-1]
    assert plan_call["model"] == "test-model"
    assert plan_call["config"].response_mime_type == "application/json"
    assert "grounded context" in plan_call["contents"]

    models.text = "  Only a grounded answer.  "
    answer = provider.generate_answer("question", "grounded context", "System 1 results")
    assert answer == "Only a grounded answer."
    answer_call = models.generate_calls[-1]
    assert "information is not available" in answer_call["config"].system_instruction
    assert "System 1 results" in answer_call["contents"]


def test_gemini_llm_failure_is_normalized():
    provider = GeminiLLMProvider(
        "invalid-key", client=FakeGeminiClient(FakeGeminiModels(error=RuntimeError("unauthorized")))
    )
    with pytest.raises(LLMProviderError, match="Gemini answer generation failed"):
        provider.generate_answer("question", "context")


@pytest.mark.parametrize("field", ["embedding_provider", "llm_provider"])
def test_unsupported_provider_configuration_is_rejected(field):
    with pytest.raises(ValidationError):
        Settings(**{field: "unsupported"})


@pytest.mark.parametrize("provider_kind", ["embedding", "llm"])
def test_provider_selector_defaults_to_openai_and_keeps_openai_available(provider_kind, monkeypatch):
    import app.embeddings.provider as embeddings
    import app.llm.provider as llm

    module = embeddings if provider_kind == "embedding" else llm
    _reset_provider_cache(module)
    monkeypatch.setattr(module, "settings", SimpleNamespace(
        embedding_provider="openai", openai_api_key="test-openai", embedding_model="test-embed",
        llm_provider="openai", llm_model="test-llm",
    ))
    get_provider = embeddings.get_embedding_provider if provider_kind == "embedding" else llm.get_llm_provider

    assert isinstance(get_provider(), OpenAIEmbeddingProvider if provider_kind == "embedding" else OpenAILLMProvider)
    _reset_provider_cache(module)


@pytest.mark.parametrize("provider_kind", ["embedding", "llm"])
def test_gemini_provider_selector_and_missing_key(provider_kind, monkeypatch):
    import app.embeddings.provider as embeddings
    import app.llm.provider as llm

    module = embeddings if provider_kind == "embedding" else llm
    _reset_provider_cache(module)
    monkeypatch.setattr(module, "settings", SimpleNamespace(
        embedding_provider="gemini", gemini_api_key="fake-gemini-key",
        gemini_embedding_model="gemini-embedding-2", openai_api_key=None,
        llm_provider="gemini", gemini_llm_model="gemini-3.8-flash", llm_model="unused",
    ))
    get_provider = embeddings.get_embedding_provider if provider_kind == "embedding" else llm.get_llm_provider
    assert isinstance(get_provider(), GeminiEmbeddingProvider if provider_kind == "embedding" else GeminiLLMProvider)

    _reset_provider_cache(module)
    monkeypatch.setattr(module, "settings", SimpleNamespace(
        embedding_provider="gemini", gemini_api_key=" ", gemini_embedding_model="gemini-embedding-2",
        llm_provider="gemini", gemini_llm_model="gemini-3.8-flash",
    ))
    assert get_provider() is None
    _reset_provider_cache(module)
