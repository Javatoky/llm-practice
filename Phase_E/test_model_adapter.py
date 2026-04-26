import asyncio

import pytest
from model_adapter import LLMBackend, ModelConfig, NotSupportError


def make_config(
    chat_model: str = "chat-model-1",
    embed_model: str | None = "text-embedding-v1",
    supports_embeddings: bool = True,
) -> ModelConfig:
    return ModelConfig(
        chat_model=chat_model,
        chat_api_key="sk-test",
        chat_base_url="http://test.local/v1",
        embed_model=embed_model,
        embed_api_key="sk-test",
        embed_base_url="http://test.local/v1",
        supports_chat=True,
        supports_embeddings=supports_embeddings,
    )


def test_config_independence():
    backend_a = LLMBackend(make_config(chat_model="chat-model-1"))
    backend_b = LLMBackend(make_config(chat_model="chat-model-2"))

    assert backend_a.config.chat_model == "chat-model-1"
    assert backend_b.config.chat_model == "chat-model-2"
    assert backend_a.config is not backend_b.config


def test_embed_error():
    backend = LLMBackend(
        make_config(
            embed_model=None,
            supports_embeddings=False,
        )
    )

    with pytest.raises(NotSupportError, match="当前配置不支持embed调用"):
        asyncio.run(backend.embed(["hello"]))