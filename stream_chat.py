from collections.abc import Generator

from openai import OpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

def chat(messages: list[dict], user_input: str) -> Generator[str, None, None]:
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        stream = True
    )
    reply = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            reply += delta
            yield delta
    messages.append({"role": "assistant", "content": reply})

messages = [
    {"role": "system", "content": "你是一个Python编程助手，回答简洁。"},
]
while True:
    print("想问什么？")
    if(user_input := input()) == "quit":
        break
    for token in chat(messages, user_input):
        print(token, end="", flush = True)
    print("\n")

print("下次再见！")