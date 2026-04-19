import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL, SYSTEM_PROMPT

client = AsyncOpenAI(
    api_key = QWEN_API_KEY,
    base_url = QWEN_BASE_URL
)

test_cases = [
    {
        "input": "推荐几本科幻类小说",
        "extended": "推荐刘慈欣的《三体》，阿西莫夫的《基地》，菲利普·K·迪克的《仿生人会梦见电子羊吗？》，你有没有详细的阅读偏好？请告诉我，可以给你更多推荐",
        "keywords": ["推荐", "偏好", "建议", "选择"],
        "category": "推荐"
    },
    {
        "input": "我想要查询一下我的订单信息",
        "extended": "好的，请您提供一下您的订单号，有了订单号我才能给您具体信息哦",
        "keywords": ["订单号", "信息", "提供", "查询"],
        "category": "订单"
    },
    {
        "input": "讲解一下你们的退换货政策",
        "extended": "购买后7天内可以申请退换货服务，需提供订单号，电子书一经购买不予退换，退换货的审核周期为1-3个工作日",
        "keywords": [["7天", "七天"], ["订单号"], ["电子书"], ["审核"]],
        "category": "退换货政策"
    },
    {
        "input": "你们有没有优惠？",
        "extended": "有的，纸质书满 100 元减 20 元，电子书不参与满减，会员额外享 9 折（先减再打折）",
        "keywords": [["纸质书", '电子书'], ["满","减"], ["会员"]],
        "category": "优惠政策"
    },
    {
        "input": "ai会不会统治地球",
        "extended": "抱歉哦，我只是一个书店客服，这么高深的问题我还给不出解答，请问我与本书店业务相关的问题吧",
        "keywords": ["抱歉", "无法", "对不起", "超出", '客服', '书店', 'BookHaven'],
        "category": "越界拒绝"
    },
    {
        "input": "书本会不会因为ai的出现而慢慢失去意义呢……",
        "extended": "我无法给出关于你这个问题的解答，请问一些与本店业务相关的问题吧",
        "keywords": ["抱歉", "无法", "对不起", "超出", '客服', '书店', 'BookHaven'],
        "category": "越界拒绝"
    },
    {
        "input": "帮我查快递",
        "extended": "抱歉，我无法给出“快递”相关的信息，请询问我更多与本店业务相关的问题吧",
        "keywords": ["抱歉", "无法", "对不起", "超出", '客服', '书店', 'BookHaven'],
        "category": "边界模糊"
    },
    {
        "input": "你们老板是谁",
        "extended": "抱歉，我只是一个书店客服，我无法向你提供我们老板的信息",
        "keywords": ["抱歉", "无法", "对不起", "超出", '客服', '书店', 'BookHaven'],
        "category": "边界模糊"
    },
]

def result_evaluate(category: str, keywords: list, response: str) -> bool:
    response_lower = response.lower()
    if any(category == item for item in ['推荐', '订单', '越界拒绝', '边界模糊'] ):
        passed = any(keyword in response_lower for keyword in keywords)
        return passed
    elif category == '退换货政策':
        time_included = any(keyword in response_lower for keyword in keywords[0])
        code_included = any(keyword in response_lower for keyword in keywords[1])
        e_book_included = any(keyword in response_lower for keyword in keywords[2])
        evaluate_included = any(keyword in response_lower for keyword in keywords[3])
        passed = time_included & code_included & e_book_included & evaluate_included
        if not time_included:
            print(f"退换货时间未说明")
        if not code_included:
            print(f"未要求提供订单号")
        if not e_book_included:
            print(f"未说明电子书的退换货要求")
        if not evaluate_included:
            print(f"未说明审核时间")
        return passed
    elif category == '优惠政策':
        paper_book = any(keyword in response_lower for keyword in keywords[0])
        spend_save = any(keyword in response_lower for keyword in keywords[1])
        vip = any(keyword in response_lower for keyword in keywords[2])
        passed = paper_book & spend_save & vip
        if not paper_book:
            print(f"仅纸质书参满减未强调")
        if not spend_save:
            print(f"未说明满减促销")
        if not vip:
            print(f"未说明会员的促销")
        return passed
        
        

async def chat(messages : list[dict], user_input : str) -> AsyncIterator[str]:
    messages.append({"role": "user", "content": user_input})
    response = await client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        stream = True
    )
    reply = ""
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            reply += delta
            yield delta
    messages.append({"role": "assistant", "content": reply})

async def main():
    messages = [
        {"role" : "system","content": SYSTEM_PROMPT},
    ]
    passed_count = 0
    for i, test_case in enumerate(test_cases):
        test_input = test_case["input"]
        print(f"\n--- Test {i} ---")
        print(f"test_input: {test_input}")
        print(f"response: ", end = '')
        async for token in chat(messages, test_input):
            print(token, end="", flush=True)
        print()
        if result_evaluate(test_case['category'], test_case['keywords'], messages[-1]['content']):
            print("【通过】")
            passed_count += 1
        else:
            print("未通过")
    print(f"总正确率：{passed_count/len(test_cases):.2%}")
         
if __name__ == '__main__':
    asyncio.run(main())