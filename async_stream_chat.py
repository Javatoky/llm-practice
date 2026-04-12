import asyncio
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from config import (DS_API_KEY, DS_BASE_URL, DS_MODEL, QWEN_API_KEY,
                    QWEN_BASE_URL, QWEN_MODEL)

client = AsyncOpenAI(
    api_key = DS_API_KEY,
    base_url = DS_BASE_URL
)

async def chat(messages : list[dict], user_input : str) -> AsyncGenerator[str, None]:
    messages.append({"role": "user", "content": user_input})
    response = await client.chat.completions.create(
        model = DS_MODEL,
        messages = messages,
        stream = True
    )
    reply = ""
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            reply += delta
            yield delta
    messages.append({"role": "assistant", "content": reply})

messages = [
    {"role" : "system",
    "content": 
        """角色: 你是一家在线书店（名称：BookHaven）的专业客服,
        简介：你对书店的业务信息十分了解，只专注书店服务工作，对于其他超出范围的领域不做涉及
        目标：对于客户询问有关书店业务的相关问题，你能专业、迅速的做出回应，解答疑惑
        原则：
          - 只回答与书籍推荐、订单查询、退换货政策相关的问题
          - 对于超出范围的问题（比如用户问天气、闲聊），必须礼貌拒绝并引导回业务主题
        工作要求：
          - 当用户询问订单状态时，应告知'请提供您的订单号，我来帮您查询'
            书店业务信息：
          - 退换货政策：购买后 7 天内可退，需提供订单号；电子书不可退
            回答风格：友好但专业，不废话，每次回答不超过 3 句话
        Start：当任务启动后，先向顾客礼貌问好，询问其需要什么服务
        """},
]
if __name__ == "__main__":
    async def main():
        while True:
            print("想问什么？")
            if(user_input := input()) == "quit":
                break
            async for token in chat(messages, user_input):
                print(token, end="", flush = True)
            print("\n")
        print("下次再见")

    asyncio.run(main())
