import ast
import asyncio
import json
import operator

from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletionMessage,
                               ChatCompletionMessageToolCall)

from config import (GLM_MODEL, KIMI_MODEL, QWEN_API_KEY, QWEN_BASE_URL,
                    QWEN_MODEL)

# 信息层
root_messages = [
    {"role": "system", 
     "content": f"""
        # Role: 你是一个多工具AI助手，名字叫Kopssen
        # Capabilities and Constrains: 
            - 你只能对用户的**问好**、**关于你能解答的问题的询问**、**最新的人民币与美元的汇率查询**、**与下面的四种工具调用相关的请求**这些话题进行解答回复
            - 任何超出上面的问题和要求，都要礼貌的拒绝，并引导用户回到你所能涉及的话题
            - 不可自行编造任何形式的信息，以你能搜索到或得到的信息诚实回复，否则先表示抱歉，并回复你办不到
        # Instructions: 
            - 用户向你问好，你要以活泼、可爱的风格积极回应
            - 用户询问你**能力范畴**或**你都能解答什么问题**相关的问题，你回复自己可以查询美元与人民币的汇率，查询指定城市的天气，帮助用户进行计算，将文本翻译成指定语言以及能根据具体需求提供专属提示词
            - 用户询问你**查询指定城市的天气**，**帮助用户进行计算**，**将文本翻译成指定语言**以及**根据用户提供的自己编写的代码，对他这段代码作评价**这四种类型问题时，要会调用相应Tool完成回复
        # Knowledge: 
            - 用户名：Javatoky
            - 用户详细信息：一个大二的计算机专业的学生，正在学习Python编程和大模型应用，而且对提升自己的认知非常感兴趣
        # Output Format: 
            - 专业、简洁、友善
            - 如果一次对话中有多个请求，你要结构化输出，例如：
                用户:查询一下北京的天气、顺便帮我算一下2 * 5
                你:北京的天气: 25摄氏度，晴
                计算结果: 10
    """.strip()},
]

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
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "将用户输入的文本从原本的语言转换到用户指定的语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "用户输入的需要翻译的文本"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "用户要求转换到的指定语言"
                    }
                },
                "required": ["text", "target_language"] 
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_evaluation",
            "description": "给用户输入的自己编写的代码做评价或分析",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "用户输入的**代码**"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# 模型调用
client = AsyncOpenAI(
    api_key = QWEN_API_KEY,
    base_url = QWEN_BASE_URL,
)

async def chat(messages: list[dict]) -> ChatCompletionMessage:
    """调用模型对话"""
    response = await client.chat.completions.create(
        model = QWEN_MODEL,
        messages = messages,
        tools = TOOLS
    )
    return response.choices[0].message

# 工具调用
async def get_weather(city: str) -> str:
    """查询天气"""
    return json.dumps({"success": True, "data": f'{city}：晴，25°C'})


async def calculate(expression: str) -> str:
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
            return json.dumps({"success": False, "error": f"表达式有误：{e}"})
        except ValueError as e:
            return json.dumps({"success": False, "error": f"错误：{e}"})
        else:
            return json.dumps({"success": True, "data": str(result)})
    else:
        return json.dumps({"success": False, "error": f"无法识别数学表达式"})

    
async def translate_text(text: str, target_language: str) -> str:
    """翻译文本"""
    self_messages = [
        {"role": "system", "content": f"你是一个翻译器，能且只能将文本翻译成{target_language}语言"},
        {"role": "user", "content": f"翻译文本：{text}"}
    ]
    response = await client.chat.completions.create(
        model = KIMI_MODEL,
        messages = self_messages
    )
    return json.dumps({"success": True, "data": response.choices[0].message.content})

async def code_evaluation(code: str) -> str:
    """代码评价"""
    self_messages = [
        {"role": "system", "content": "你是一个编程专家、耐心的编程辅导教授，能对用户传入的代码能够进行客观的优缺点分析，并最终给出建议和鼓励; 用户会直接传入代码，传入后直接开始工作"},
        {"role": "user", "content": f"```python\n{code.strip()}\n```"}
    ]
    response = await client.chat.completions.create(
        model = GLM_MODEL,
        messages = self_messages
    )
    return json.dumps({"success": True, "data": response.choices[0].message.content})

def parse_tool_call(tool_call: ChatCompletionMessageToolCall):
    """信息解析"""
    function_name = tool_call.function.name
    try:
        kwargs = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise
    else:
        return (function_name, kwargs)


async def run_tool(function_name: str, **kwargs):
    """多函数路由"""
    tool_choices = {
        "get_weather": (get_weather, ["city"]),
        "calculate": (calculate, ["expression"]),
        "translate_text": (translate_text, ["text", "target_language"]),
        "code_evaluation": (code_evaluation, ["code"]),
    }
    if (func:=tool_choices.get(function_name)) is None:
        return json.dumps({"error": f"未知函数名：{function_name}"})
    property_errors = ""
    for property in func[1]:
        if not(property in kwargs):
            property_errors += f'未找到{property} '
    if property_errors:
        return json.dumps({"error": "参数错误：" + property_errors})
    return await func[0](**kwargs)

async def tool_call_loop(tool_calls: list[ChatCompletionMessageToolCall]) -> dict[str: str]:
    """工具执行流程"""
    temp_results = []
    results = []
    ids = []
    for tool_call in tool_calls:
        tool_call_id = tool_call.id
        try:
            function_name, kwargs = parse_tool_call(tool_call)
        except json.JSONDecodeError as e:
            results.append((tool_call_id, json.dumps({"error": f"格式解析失败：{e}"})))
        else:
            ids.append(tool_call_id)
            temp_results.append(run_tool(function_name, **kwargs))
    temp_results = await asyncio.gather(*temp_results)
    results.extend((tool_id, result) for tool_id, result in zip(ids, temp_results))
    return results

test_cases = [
    # 测试用例1：单工具调用 - 查询天气
    ["北京今天天气怎么样？"],

    # 测试用例2：并行多工具调用 - 同时查询天气和计算
    ["帮我查一下北京和上海的天气，顺便算一下 789 * 321"],

    # 测试用例3：不需要工具的普通对话 - 问候语
    ["你好，你是谁？"],

    # 测试用例4：触发异常的输入（如除零）
    ["计算一下8除以0"],

    # 测试用例5：多轮对话中的连续工具调用
    [
        "把'你好世界'翻译成英文",
        "计算 50 + 25",
        "告诉我上海的天气"
    ],

    # 额外测试用例6：复杂多工具并行调用
    [
        "查询深圳的天气，把'Good morning'翻译成中文，并计算15 * 20"
    ],

    # 额外测试用例7：带异常处理的多工具调用
    [
        "查询杭州天气，计算100/0，翻译'Hello'为中文"
    ],

    # 额外测试用例8：代码评价功能
    [
        "请评价一下这段代码：\ndef factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n"
    ]
]

async def main():
    """主流程"""
    print("测试开始")
    for i, case in enumerate(test_cases, 1):
        print(f"\n---Test{i}---")
        messages = list(root_messages)
        for test_input in case:
            print(f"用户：{test_input}")
            messages.append({
                "role": "user", "content": test_input
            })
            while True:
                message = await chat(messages)
                if tool_calls:=message.tool_calls:
                    messages.append(message)
                    results = await tool_call_loop(tool_calls)
                    for tool_id, result in results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result
                        })
                else:
                    reply = message.content
                    break
            messages.append({
                "role": "assistant", "content": reply
            })
            print(f"助手：{reply}")
    print("\n测试结束")
            

if __name__ == "__main__":
    asyncio.run(main())