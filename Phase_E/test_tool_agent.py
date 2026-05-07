import asyncio
import json
import sys
from pathlib import Path

import pytest
from model_adapter import LLMBackend, ModelConfig
from openai.types.chat import (ChatCompletionMessage,
                               ChatCompletionMessageToolCall)
from tool_agent import (ArgumentError, ToolNameError, calculate, get_weather,
                        invoke, parse_tool_call, resolve_tool, tool_call_loop)

sys.path.insert(0, str(Path(__file__).parent.parent))


from config import (QWEN_ALY_API_KEY, QWEN_API_KEY, QWEN_BASE_URL,
                    QWEN_EMBED_MODEL, QWEN_MODEL)


def create_tool_call(
        tool_call_id : str | None = "call_abc123",
        func_name : str | None = "func",
        arguments : str | None  = json.dumps({"property": "123"}),
) -> ChatCompletionMessageToolCall:
    tool_call = ChatCompletionMessageToolCall(
        id = tool_call_id,
        function = {"name": func_name, "arguments": arguments},
        type = "function"
    )
    return tool_call


class FakeBackend:
    def __init__(self, message_to_return):
        self.message_to_return = message_to_return
        self.index = 0

    async def chat_message(self, message:list[dict]) -> ChatCompletionMessage:
        message = self.message_to_return[self.index]
        self.index += 1
        return message


def test_parse_tool_call(): 
    tool_call_a = create_tool_call(
        func_name = "get_weather",
        arguments = json.dumps({"city": "北京"})
    )
    tool_call_b = create_tool_call(
        tool_call_id = "call_abc_456",
        func_name = "calculate",
        arguments = json.dumps({"expression": "1 + 1"})
    )
    wrong_tool_call = create_tool_call(
        tool_call_id = "call_abc_789",
        func_name = "bad_call",
        arguments = "{'name': 'Alice'}"
    )

    assert parse_tool_call(tool_call_a) == ("get_weather", {"city": "北京"})
    assert parse_tool_call(tool_call_b) == ("calculate", {"expression": "1 + 1"})

    with pytest.raises(json.JSONDecodeError):
        parse_tool_call(wrong_tool_call)


def test_route_tool():
    func1 = ("get_weather", {"city": "北京"})
    func2 = ("calculate", {"expression": "1 + 1"})
    wrong_func1 = ("embed", {"texts": ["你是谁", "我能成为世界上最厉害的工程师吗"]})
    wrong_func2 = ("get_weather", {"text": "如何写一份完美的提示词", "chunk_size": 300, "overlap": 50})
    wrong_func3 = ("calculate", {"expression": "1 * 4.6", "type": "float"})

    assert resolve_tool(*func1) == get_weather
    assert resolve_tool(*func2) == calculate

    with pytest.raises(ToolNameError, match = "未知工具名：embed"):
        resolve_tool(*wrong_func1)
    
    with pytest.raises(ArgumentError, match = "参数缺失：{'city'}"):
        resolve_tool(*wrong_func2)

    with pytest.raises(ArgumentError, match = "存在多余参数：{'type'}"):
        resolve_tool(*wrong_func3)

def test_invoke():
    func1 = (get_weather, {"city": "上海"})
    func2 = (calculate, {"expression": "1 + 1"})
    wrong_func1 = (calculate, {"expression": "__import__('os').system('whoami')"})
    wrong_func2 = (calculate, {"expression": "9 / 0"})
    wrong_func3 = (calculate, {"expression": ""})

    assert asyncio.run(invoke(*func1)) == {"success": True, "data": f'{"上海"}：晴，25°C'}
    assert asyncio.run(invoke(*func2)) == {"success": True, "data": str(2)}

    wrong_result1 = asyncio.run(invoke(*wrong_func1))
    wrong_result2 = asyncio.run(invoke(*wrong_func2))
    wrong_result3 = asyncio.run(invoke(*wrong_func3))

    assert wrong_result1["success"] == False and "不支持的表达式" in wrong_result1["error"]
    assert wrong_result2["success"] == False and "表达式有误" in wrong_result2["error"]
    assert wrong_result3["success"] == False and "无法识别数学表达式" in wrong_result3["error"]


# TODO(stage-e): add FakeBackend tests for tool_call_pipeline and tool_call_loop before final integration.


TEST_CASES = [
    "北京今天天气怎么样？",
    "帮我算一下 789 * 321"
]

root_prompt = f"""
    # Role: 你是一个多工具AI助手，名字叫Kopssen
    # Capabilities and Constrains: 
        - 你只能对用户的**问好**、**关于你能解答的问题的询问**、**与下面的两种工具调用相关的请求**这些话题进行解答回复
        - 任何超出上面的问题和要求，都要礼貌的拒绝，并引导用户回到你所能涉及的话题
        - 不可自行编造任何形式的信息，以你调用工具得到的信息诚实回复，否则先表示抱歉，并回复你办不到
    # Instructions: 
        - 用户向你问好，你要以活泼、可爱的风格积极回应
        - 用户询问你**能力范畴**或**你都能解答什么问题**相关的问题，你回复自己可以查询指定城市的天气，帮助用户进行计算
        - 用户询问你**查询指定城市的天气**，**帮助用户进行计算**，这两种类型问题时，要会调用相应Tool完成回复
        - 如果Tool返回 success=false ， 请向用户说明失败原因，不得编造
    # Knowledge: 
        - 用户名：Javatoky
        - 用户详细信息：一个大二的计算机专业的学生，正在学习Python编程和大模型应用，而且对提升自己的认知非常感兴趣
    # Output Format: 
        - 专业、简洁、友善
        - 如果一次对话中有多个请求，你要结构化输出，例如：
            用户:查询一下北京的天气、顺便帮我算一下2 * 5
            你:北京的天气: 25摄氏度，晴
            计算结果: 10
""".strip()


async def main():
    backend = LLMBackend(ModelConfig(QWEN_MODEL, QWEN_API_KEY, QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_ALY_API_KEY, QWEN_BASE_URL, True, True))

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n---Test{i}---")
        messages = [
            {"role": "system", "content": root_prompt}
        ]
        
        messages.append({
            "role": "user", "content": case
        })
        reply = await tool_call_loop(backend, messages)

        print("上下文内容追踪：")
        for message in messages:
            print(message)

        print("\n最终回复：")
        print(reply)

if __name__ == "__main__":
    asyncio.run(main())