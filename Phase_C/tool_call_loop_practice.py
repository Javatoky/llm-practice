import json

from openai import OpenAI

from config import KIMI_MODEL, QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL, TOOLS

# 大模型调用
client = OpenAI(
    api_key = QWEN_API_KEY,
    base_url = QWEN_BASE_URL,
)

def chat(messages: list[dict]):
    response = client.chat.completions.create(
        model = QWEN_MODEL,
        messages = messages,
        tools = TOOLS
    )
    return response.choices[0].message

# 函数的执行，包括tool_call解析、函数选择、所用函数
def parse_tool_call(tool_call: dict):
    function_name = tool_call.function.name
    try:
        kwargs = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        raise
    
    return (function_name, kwargs)

def run_tool_call(function_name, **kwargs):     # 我的思考：在TOOLS中我们明确了各个函数的参数名，LLM传回一定会是以字典形式参数名与值一一对应，所以要解包字典获取参数
    tool_choices = {
        "get_weather": get_weather,
        "calculate": calculate,
        "translate_text": translate_text
    }
    return tool_choices[function_name](**kwargs)

def get_weather(city: str) -> str:
    return f'{city}：晴，25°C'

def calculate(expression: str) -> str:
    if expression:
        return str(eval(expression))
    else:
        return f"无法识别数学表达式"
    
def translate_text(text: str, target_language: str) -> str:
    messages = [
        {"role": "system", "content": f"你是一个翻译器，能且只能将文本翻译成{target_language}"},
        {"role": "user", "content": f"翻译文本：{text}"}
    ]
    response = client.chat.completions.create(
        model = KIMI_MODEL,
        messages = messages
    )
    return response.choices[0].message.content

test_cases = [
    "北京今天天气怎么样？",
    "帮我算一下 1234 乘以 5678",
    "把'今天天气真好'翻译成英文",
    "计算一下8除以0"
]

# 主流程
def main():
    print("---测试开始---")
    for i,case in enumerate(test_cases, start = 1):
        print(f"\n---Test{i}---")
        messages = [
            {"role": "system", "content": "你是一个智能助手，能根据用户需求，选择是否调用Tool，从而解决问题"},
            {"role": "user", "content": case}
        ]
        while True:
            message = chat(messages)
            if tool_calls := message.tool_calls:
                messages.append({
                    "role": "assistant", "tool_calls": tool_calls
                })
                for tool_call in tool_calls:
                    tool_call_id = tool_call.id
                    function_name, kwargs = parse_tool_call(tool_call)
                    result = run_tool_call(function_name, **kwargs)
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
    main()