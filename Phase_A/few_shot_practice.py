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
    "推荐几本言情小说",
    "我想查一下我的订单到哪了",
    "你们的退换货政策是什么？",
    "给我做一首诗歌",
    "你觉得人工智能会取代人类吗？",
]

async def run_tests() -> None:
    """每个测试用例独立运行，不共享对话历史。"""
    for i, test_input in enumerate(TEST_CASES, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "向我推荐科幻小说"},
            {"role": "assistant", "content": "推荐小说《三体》，适合科幻爱好者，你可以告诉我更多有关你的阅读偏好以获取更多推荐。"},
            {"role": "user", "content": "查询一下今天天气"},
            {"role": "assistant", "content": "抱歉，天气查询不是我的业务范围哦，请问我与本书店业务相关的问题。"},
            {"role": "user", "content": "我要退换货"},
            {"role": "assistant", "content": "请您需提供订单号（购买后7天内可申请退换货）。电子书不支持退换，退换货审核周期为 1-3 个工作日。"},
        ]
        print(f"\n--- Test {i} ---")
        print(f"User: {test_input}")
        print("Assistant: ", end="")
        async for token in chat_stream(messages, test_input):
            print(token, end="", flush=True)
        print("\n")


asyncio.run(run_tests())