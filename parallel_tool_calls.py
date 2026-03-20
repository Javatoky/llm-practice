import asyncio
import json

from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletionMessage,
                               ChatCompletionMessageToolCall)

from config import KIMI_MODEL, QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL, TOOLS


class FunctionNotFound(Exception):
    def __init__(self, function_name: str):
        self.function_name = function_name
        super().__init__(f"未知的函数: {function_name}")

class PropertiesInvalidError(Exception):
    def __init__(self, property: str):
        self.property = property
        super().__init__(f"参数{property}未找到")

# 大模型调用
client = AsyncOpenAI(
    api_key = QWEN_API_KEY,
    base_url = QWEN_BASE_URL,
)

async def chat(messages: list[dict]) -> ChatCompletionMessage:
    response = await client.chat.completions.create(
        model = QWEN_MODEL,
        messages = messages,
        tools = TOOLS
    )
    return response.choices[0].message

# 函数的执行，包括tool_call解析、函数选择、所用函数
def parse_tool_call(tool_call: ChatCompletionMessageToolCall):
    """解析tool_call"""
    function_name = tool_call.function.name
    try:
        kwargs = json.loads(tool_call.function.arguments)       # 大模型传入的数据不能保证其严格遵循可转化的JSON格式，所以要严格检查
    except json.JSONDecodeError as e:
        raise e
    else: 
        return (function_name, kwargs)

async def run_tool_call(function_name: str, **kwargs):     # 我的思考：在TOOLS中我们明确了各个函数的参数名，LLM传回一定会是以字典形式参数名与值一一对应，所以要解包字典获取参数
    """函数路由：实现函数映射和名称、参数名的检查"""
    tool_choices = {
        "get_weather": (get_weather, ["city"]),
        "calculate": (calculate, ["expression"]),
        "translate_text": (translate_text, ["text", "target_language"])
    }
    func_and_properties = tool_choices.get(function_name)
    if not func_and_properties:
        return f"{FunctionNotFound(function_name)}"
    for property in func_and_properties[1]:
        if not kwargs.get(property):
            return f"{PropertiesInvalidError(property)}"
    try:
        result = await func_and_properties[0](**kwargs)
    except Exception as e:
        return f"函数执行失败：{e}"
    else:
        return result

async def get_weather(city: str) -> str:
    # 思考：有没有什么办法可以检查输入的city一定是城市？
    return f'{city}：晴，25°C'

async def calculate(expression: str) -> str:
    # TODO:解决eval计算的安全隐患问题
    if expression:
        try:
            result = str(eval(expression))
        except ZeroDivisionError:
            return f"错误：除数不能为零！"
        except ValueError as e:
            return f"错误：{e}"
        else:
            return result
    else:
        return f"错误：无法识别数学表达式"
    
async def translate_text(text: str, target_language: str) -> str:
    # 思考：对于这个函数，除了提示词提前约束，有没有什么办法确定target_language一定是“语言”，避免幻觉输入？
    messages = [
        {"role": "system", "content": f"你是一个翻译器，能且只能将文本翻译成{target_language}语言"},
        {"role": "user", "content": f"翻译文本：{text}"}
    ]
    response = await client.chat.completions.create(
        model = KIMI_MODEL,
        messages = messages
    )
    return response.choices[0].message.content

test_cases = [
    "北京今天天气怎么样？",
    "帮我查一下北京和上海的天气，顺便算一下 789 * 321",
    "把'你好世界'翻译成英文和日文",
    "计算一下8除以0"
]

# 主流程
async def main():
    # 思考：我的主流程逻辑是否还可以优化一下
    print("---测试开始---")
    for i,case in enumerate(test_cases, start = 1):
        print(f"\n---Test{i}---")
        messages = [
            {"role": "system", "content": "你是一个智能助手，能根据用户需求，选择是否调用Tool，从而解决问题"},
            {"role": "user", "content": case}
        ]
        while True:
            message = await chat(messages)
            if tool_calls := message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls]
                })
                results = []
                ids = []
                for tool_call in tool_calls:
                    tool_call_id = tool_call.id
                    try:
                        function_name, kwargs = parse_tool_call(tool_call)
                    except json.JSONDecodeError as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id, 
                            "content": f"错误：{e}"
                        })
                        continue
                    results.append(run_tool_call(function_name, **kwargs))
                    ids.append(tool_call_id)
                results = await asyncio.gather(*results)
                for tool_call_id, result in zip(ids, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id, 
                        "content": result
                    })
            else:
                reply = message.content
                break
        messages.append({
            "role": "assistant", "content": reply
        })
        print(reply)
    print("\n---测试结束---")

if __name__ == "__main__":
    asyncio.run(main())