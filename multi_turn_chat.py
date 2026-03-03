from openai import OpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

def chat(messages: list[dict], user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

messages = [
    {"role": "system", "content": "你是一个Python编程助手，回答简洁。"},
]
while True:
    print("想问什么？")
    if(user_input := input()) == "quit":
        break
    print(chat(messages, user_input))
print("下次再见！")
