import ast
import json
import asyncio
import operator
from typing import Callable
from model_adapter import LLMBackend
from openai.types.chat import ChatCompletionMessageToolCall


class ToolNameError(Exception):
    """工具名异常"""
    pass


class ArgumentError(Exception):
    """参数异常"""
    pass


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当天天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "用户输入的城市名称，如'上海'或'Shanghai'、'北京'或'Beijing'"
                    }
                },
                "required": ["city"]
            }   
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算用户输入的数学表达式，返回计算结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression":{
                        "type": "string",
                        "description": "输入的数学表达式，如'1234 * 5678'、'67 - 34'；如果用户输入的字符串不能明显理解成一个数学表达式或其数学表达式非法（比如除数为零），必须将此参数设为空字符"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


async def get_weather(city: str) -> dict:
    """查询天气"""
    return {"success": True, "data": f'{city}：晴，25°C'}


async def calculate(expression: str) -> dict:
    """计算表达式"""
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    def safe_eval(expr: str) -> float:
        """只允许基本四则运算的安全表达式求值。"""
        tree = ast.parse(expr, mode="eval")
        
        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
                return SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
            else:
                raise ValueError(f"不支持的表达式: {ast.dump(node)}")
        
        return _eval(tree)
    if expression:
        try:
            result = safe_eval(expression)
        except ZeroDivisionError as e:
            return {"success": False, "error": f"表达式有误：{e}"}
        except ValueError as e:
            return {"success": False, "error": f"错误：{e}"}
        else:
            return {"success": True, "data": str(result)}
    else:
        return {"success": False, "error": f"无法识别数学表达式"}


def parse_tool_call(tool_call: ChatCompletionMessageToolCall) -> tuple[str, dict]:
    """工具请求解析"""
    func_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        raise

    return (func_name, arguments)


def route_tool(func_name: str, arguments: dict) -> Callable:
    """工具路由"""
    tool_config = {
        "get_weather": (get_weather, ["city"]),
        "calculate": (calculate, ["expression"])
    }

    if (tool_info:=tool_config.get(func_name, None)) is None:
        raise ToolNameError(f"未知工具名：{func_name}")
    argument_errors = ""
    for argument in tool_info[1]:
        if argument not in arguments:
            argument_errors += f'未找到参数：{argument};'
    if argument_errors:
        raise ArgumentError(f"{argument_errors}")
    
    return tool_info[0]


async def invoke(tool: Callable, arguments: dict) -> dict:
    """调用器"""
    return await tool(**arguments)


async def tool_call_pipeline(tool_calls: list[ChatCompletionMessageToolCall]) -> list[tuple[str, dict]]:
    """工具调用流"""
    results = []
    ids = []
    temp_results = []

    for tool_call in tool_calls:
        tool_call_id = tool_call.id
        try:
            func_name, arguments = parse_tool_call(tool_call)
            tool = route_tool(func_name, arguments)
        except json.JSONDecodeError as e:
            results.append((tool_call_id, {"error": f"格式解析失败：{e}"}))
        except ToolNameError as e:
            results.append((tool_call_id, {"error": f"工具名错误：{e}"}))
        except ArgumentError as e:
            results.append((tool_call_id, {"error": f"参数错误：{e}"}))
        else:
            ids.append(tool_call_id)
            temp_results.append(invoke(tool, arguments))
    
    temp_results = await asyncio.gather(*temp_results)
    results.extend(zip(ids, temp_results))

    return results


async def tool_call_loop(backend: LLMBackend, messages: list[dict]) -> str:
    """工具调用循环"""
    while True:
        message = await backend.chat_message(messages, tools = TOOLS)
        
        if (tool_calls:=message.tool_calls) is not None:
            messages.append(message)
            results = await tool_call_pipeline(tool_calls)
            for id, result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": id,
                    "result": json.dumps(result)
                })
        elif (content:=message.content) is not None:
            messages.append(
                {"role": "assistant", "content": content}
            )

            return content