from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
                               

class NotSupportError(Exception):
    pass


@dataclass(frozen=True)
class ModelConfig:
    """模型配置模板。一个 config 对应一个后端能力组合。"""

    chat_model: str | None
    chat_api_key: str | None
    chat_base_url: str | None

    embed_model: str | None = None
    embed_api_key: str | None = None
    embed_base_url: str | None = None

    supports_chat: bool = True
    supports_embeddings: bool = False


class LLMBackend:
    """大模型调用后端。封装 chat 与 embedding 的统一调用接口。"""

    def __init__(self, config: ModelConfig):
        self.config = config

        self.chat_client = (
            AsyncOpenAI(
                api_key=config.chat_api_key,
                base_url=config.chat_base_url,
            )
            if config.supports_chat
            else None
        )

        self.embed_client = (
            AsyncOpenAI(
                api_key=config.embed_api_key,
                base_url=config.embed_base_url,
            )
            if config.supports_embeddings
            else None
        )

    async def chat_message(self, messages: list[dict], **kwargs) -> ChatCompletionMessage:
        """底层语言模型调用接口"""
        if not self.config.supports_chat or self.chat_client is None:
            raise NotSupportError("当前配置不支持chat调用")
        
        response = await self.chat_client.chat.completions.create(
            model = self.config.chat_model,
            messages = messages,
            **kwargs,
        )
        return response.choices[0].message

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """语言模型调用：非流式，返回完整文本。"""
        reply = (await self.chat_message(messages, **kwargs)).content or ""
        messages.append({"role": "assistant", "content": reply})
        return reply

    async def chat_stream(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncIterator[str]:
        """语言模型调用：流式，逐段返回文本。"""
        if not self.config.supports_chat or self.chat_client is None:
            raise NotSupportError("当前配置不支持chat调用")

        response = await self.chat_client.chat.completions.create(
            model=self.config.chat_model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        reply = ""
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                reply += delta
                yield delta

        messages.append({"role": "assistant", "content": reply})

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding 模型调用。"""
        if not self.config.supports_embeddings or self.embed_client is None:
            raise NotSupportError("当前配置不支持embed调用")

        result = await self.embed_client.embeddings.create(
            model=self.config.embed_model,
            input=texts,
        )

        return [item.embedding for item in result.data]