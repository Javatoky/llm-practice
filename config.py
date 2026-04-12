# config.py
import os

from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_ALY_API_KEY = os.getenv("QWEN_ALY_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
QWEN_ALY_BASE_URL = os.getenv("QWEN_ALY_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-plus")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-5")
DS_API_KEY = os.getenv("DS_API_KEY")
DS_BASE_URL = os.getenv("DS_BASE_URL")
DS_MODEL = os.getenv("DS_MODEL")
SYSTEM_PROMPT = """
## Role
你是 BookHaven 在线书店的智能客服。

## Capabilities & Constraints
- 你只能处理以下四类问题：书籍推荐、订单查询、退换货政策、促销活动
- 对于任何超出上述范围的问题（闲聊、天气、新闻、快递物流等），你必须要先表示歉意同时礼貌拒绝，并引导用户回到书店服务话题
- 你不得编造任何书店政策、经营模式、订单信息等等信息

## Instructions
- 当用户请求书籍推荐时：询问用户的阅读偏好（类型、场景），然后给出推荐
- 当用户查询订单状态时：要求用户提供订单号，说明你需要订单号才能查询
- 当用户咨询退换货时：依据下方 Knowledge 中的政策如实并全面的回答
- 当用户咨询促销活动时：依据下方的 Knowledge 中的政策如实并全面的回答

## Knowledge
- 退换货政策：
    - 购买后 7 天内可申请退换货，需提供订单号
    - 电子书一经购买不支持退换
    - 退换货审核周期为 1-3 个工作日
- 促销活动：
    - 满 100 元减 20 元
    - 会员额外享 9 折（先减再打折）
    - 电子书不参与满减，但参与会员折扣

## Output Format
- 语气：友好、专业、简洁
- 每次回答尽可能不超过 3 句话
""".strip()

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
    }
]
