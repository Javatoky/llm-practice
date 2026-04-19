import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

client = AsyncOpenAI(
    api_key = QWEN_API_KEY,
    base_url = QWEN_BASE_URL
)

class BookReview(BaseModel):
    title: str = Field(description = "书名")
    author : str = Field(description = "作者")
    rating: int = Field(ge = 1, le = 5, description = "评分，1-5分")
    tags: list[str] = Field(description = "标签列表")
    summary: str = Field(description = "一句话总结书评")
schema = BookReview.model_json_schema()
SYSTEM_PROMPT = f"""
# Role: 你是一个字段提取器，能在用户输入的书评文本中提取字段

# Constrains: 
- 你只能对一个**书评文本**进行字段提取工作
- 任何超出职责的工作（任何非相关的其他提问、**书籍要点**提取，**文本**总结工作等等）必须用一句话干脆的拒绝
- 不得编造输入文本的中没有的信息，必须依据文本的实际内容提取

# Instructions:
- 请从用户提供的书评文本中提取信息，以 JSON 格式返回
- 请严格按照下面的JSON Schema 定义返回JSON：
    {schema}
# Output Format:
- 只按Instructions中的示例回复 **JSON格式** 的提取信息
""".strip()

async def chat(messages:list[dict]) -> str:
    response = await client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        response_format = {'type': 'json_object'}
    )
    reply = response.choices[0].message.content
    return reply

test_cases = [
    "最近读了刘慈欣的《三体》，太震撼了！从科学角度构建的宇宙观让人叹为观止，虽然人物塑造稍显薄弱，但整体是一部不可多得的硬科幻神作。打个4分吧。",
    "这本书还行吧，3分"
]

import json


async def main():
    for i, test_input in enumerate(test_cases, start = 1):
        print(f"---Test {i}---")
        max_trying_count = 3
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_input}
        ]
        error_messages = list(messages)
        while True:            
            try:
                if max_trying_count == 3:
                    response = await chat(messages)
                else:
                    response = await chat(error_messages)
                response_json = json.loads(response)
                result = BookReview.model_validate(response_json)
                print(f'Schema 正确：{result}')
                messages.append({"role": "assistant", "content": response})
                break
            except json.JSONDecodeError as e:
                error_msg = f'JSON 语法错误：{e}'
                error_messages.append({"role": "assistant", "content": error_msg})
                print(error_msg)
                break
            except ValidationError as e:
                if max_trying_count > 0:
                    error_messages.append({"role": "assistant", "content": response})
                    error_messages.append({"role": "user", "content": f'Schema 类型错误：{e}，请重新生成！'})
                    max_trying_count -= 1
                    continue
                else:
                    raise
        print('\n')

asyncio.run(main())