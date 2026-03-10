SYSTEM_PROMPT = """
# Role: 你是一个字段提取器，能在用户输入的书评文本中提取字段

# Constrains: 
- 你只能对一个**书评文本**进行字段提取工作
- 任何超出职责的工作（任何非相关的其他提问、**书籍要点**提取，**文本**总结工作等等）必须用一句话干脆的拒绝
- 不得编造输入文本的中没有的信息，必须依据文本的实际内容提取

# Instructions:
- 请从用户提供的书评文本中提取信息，以 JSON 格式返回
- 输出示例：
    {
        title (string): '自私的基因'
        author (string): '理查德·道金斯'
        rating (number): 4
        tags (list[string]): ["生物学", "科普"]
        summary (string): 用基因的视角重解生命与人性
    }
# Output Format:
- 只按Instructions中的示例回复 **JSON格式** 的提取信息
"""
import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

client = AsyncOpenAI(
    base_url = QWEN_BASE_URL,
    api_key = QWEN_API_KEY
)

async def chat(messages:list, user_input: str) -> AsyncIterator[str]:
    messages.append({"role": 'user', 'content': user_input})
    response = await client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        stream = True,
        response_format = {'type': 'json_object'}
    )
    reply = ''
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            reply += delta
            yield delta
    messages.append({'role': 'assistant', 'content': reply})

test_cases = [
    "最近读了刘慈欣的《三体》，太震撼了！从科学角度构建的宇宙观让人叹为观止，虽然人物塑造稍显薄弱，但整体是一部不可多得的硬科幻神作。打个4分吧。",
]

import json


async def main():
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
    ]
    for test_input in test_cases:
        response = ''
        if test_input :
            print(f'user: {test_input}')
            async for token in chat(messages, test_input):
                response += token
            print(f'assistant: ', end = '')
            print(json.loads(response))

asyncio.run(main())