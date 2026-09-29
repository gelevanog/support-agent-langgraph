"""Provider wiring: every LLM_PROVIDER / EMBEDDINGS_PROVIDER builds a correctly configured client.

No network: the clients are only constructed. Real calls are exercised by the eval suite with
your own keys (see README > Evaluation).
"""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from support_agent.config import LLMProvider, Settings
from support_agent.knowledge import build_embedding_model
from support_agent.llm import FakeSupportModel, build_chat_model, model_label
from support_agent.llm.factory import OPENROUTER_HEADERS


def settings(**overrides: object) -> Settings:
    keys = {"openai_api_key": "sk-test", "anthropic_api_key": "sk-ant-test", "openrouter_api_key": "sk-or-test"}
    return Settings(_env_file=None, **keys, **overrides)  # type: ignore[call-arg, arg-type]


def test_openrouter_uses_the_openai_client_with_openrouter_endpoint() -> None:
    model = build_chat_model(settings(llm_provider="openrouter", openrouter_model="anthropic/claude-sonnet-5"))
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "anthropic/claude-sonnet-5"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "sk-or-test"  # not the OpenAI key
    assert model.use_responses_api is False  # OpenRouter's stable surface is Chat Completions
    assert model.default_headers == OPENROUTER_HEADERS
    assert model.max_tokens == 8192


def test_openrouter_base_url_is_configurable() -> None:
    model = build_chat_model(settings(llm_provider="openrouter", openrouter_base_url="http://gateway.local/v1"))
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "http://gateway.local/v1"


@pytest.mark.parametrize(
    ("provider", "cls", "label"),
    [
        ("fake", FakeSupportModel, "fake"),
        ("openai", ChatOpenAI, "openai/gpt-5-mini"),
        ("anthropic", ChatAnthropic, "anthropic/claude-sonnet-5"),
        ("openrouter", ChatOpenAI, "openrouter/openai/gpt-5.4-mini"),
    ],
)
def test_every_provider_builds_and_is_labelled(provider: LLMProvider, cls: type, label: str) -> None:
    configured = settings(llm_provider=provider)
    assert isinstance(build_chat_model(configured), cls)
    assert model_label(configured) == label


def test_openrouter_embeddings_send_plain_text_to_openrouter() -> None:
    built = build_embedding_model(settings(embeddings_provider="openrouter", embeddings_dimensions=512))
    assert (built.name, built.dimensions) == ("openrouter/openai/text-embedding-3-small", 512)
    embeddings = built.embeddings
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "openai/text-embedding-3-small"
    assert embeddings.openai_api_base == "https://openrouter.ai/api/v1"
    assert embeddings.check_embedding_ctx_length is False  # tiktoken token ids are OpenAI-only
    assert embeddings.dimensions == 512
