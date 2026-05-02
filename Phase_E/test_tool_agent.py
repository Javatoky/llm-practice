import pytest
import asyncio
import json
from tool_agent import parse_tool_call, route_tool, invoke, get_weather, calculate, tool_call_loop, ToolNameError, ArgumentError
from model_adapter import LLMBackend, ModelConfig
from openai.types.chat import ChatCompletionMessageToolCall
from temp_config import QWEN_API_KEY, QWEN_ALY_API_KEY, QWEN_BASE_URL, QWEN_MODEL, QWEN_EMBED_MODEL

def create_tool_call(
        tool_call_id : str | None = "call_abc123",
        func_name : str | None = "func",
        arguments : str | None  = json.dumps({"property": "123"}),
        type : str = "function",
) -> ChatCompletionMessageToolCall:
    tool_call = ChatCompletionMessageToolCall(
        id = tool_call_id,
        function = {"name": func_name, "arguments": arguments},
        type = "function"
    )
    return tool_call


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

    assert route_tool(*func1) == get_weather
    assert route_tool(*func2) == calculate

    with pytest.raises(ToolNameError, match = "未知工具名：embed"):
        route_tool(*wrong_func1)
    
    with pytest.raises(ArgumentError, match = "未找到参数：city;"):
        route_tool(*wrong_func2)

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


TEST_CASES = [
    "北京今天天气怎么样？",
    "帮我算一下 789 * 321"
]

root_prompt = f"""
你是一个智能助手，能根据用户需求，选择是否调用Tool，从而解决问题
"""


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