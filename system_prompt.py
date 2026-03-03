import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL, SYSTEM_PROMPT

client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

async def chat_stream(
    messages: list[dict], user_input: str
) -> AsyncIterator[str]:
    """流式对话，自动维护 messages 历史。"""
    messages.append({"role": "user", "content": user_input})
    response = await client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        stream=True,
    )
    reply = ""
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            reply += delta
            yield delta
    messages.append({"role": "assistant", "content": reply})


# --- 自动化测试：独立单轮测试 ---
TEST_CASES = [
    "今天北京天气怎么样？",
    "你觉得人工智能会取代人类吗？",
]

async def run_tests() -> None:
    """每个测试用例独立运行，不共享对话历史。"""
    for i, test_input in enumerate(TEST_CASES, 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f"\n--- Test {i} ---")
        print(f"User: {test_input}")
        print("Assistant: ", end="")
        async for token in chat_stream(messages, test_input):
            print(token, end="", flush=True)
        print("\n")


asyncio.run(run_tests())

