from openai import OpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

response = client.chat.completions.create(
    model=QWEN_MODEL,
    messages=[
        {"role": "system", "content": "你是一个Python编程助手，回答简洁。"},
        {"role": "user", "content": "用一句话解释什么是生成器"},
    ],
)

print(response)