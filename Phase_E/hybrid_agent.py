import copy

from agent_loop import tool_call_loop
from debug_config import AgentResult
from model_adapter import LLMBackend
from rag_agent import retrieve
from tool_agent import calculate, get_weather

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
            "name": "search_documents",
            "description": "当用户的问题需要基于本地知识库、上传文档、资料内容、笔记或参考文本回答时，检索相关文档片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索本地知识库的查询语句，通常是用户原始问题或其改写版本。"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def make_search_documents_tool(backend: LLMBackend, collection_name: str):
    async def search_documents(query: str) -> dict:
        results = await retrieve(backend, query, collection_name)
        if not results:
            return {"success": False, "error": "知识库中没有找到相关资料"}
        return {"success": True, "data": results}
    
    return search_documents

async def hybrid_agent_reply(
    backend: LLMBackend,
    user_input: str,
    messages: list[dict],
    collection_name: str,
    debug: bool = False
) -> str | AgentResult:
    tool_registry = {
        "get_weather": (get_weather, ["city"]),
        "calculate": (calculate, ["expression"]),
        "search_documents": (make_search_documents_tool(backend, collection_name), ["query"])
    }

    temp_messages = copy.deepcopy(messages)
    temp_messages.append({"role": "user", "content": user_input})
    answer = await tool_call_loop(backend, temp_messages, TOOLS, tool_registry, debug)

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": answer})
    
    return answer