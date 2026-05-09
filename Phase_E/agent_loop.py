import json
from typing import Callable

from debug_config import AgentResult
from model_adapter import LLMBackend
from tool_agent import (TOOL_REGISTRY, TOOLS, build_assistant_message,
                        tool_call_pipeline)


async def tool_call_loop(
    backend: LLMBackend,
    messages: list[dict],
    tools: list[dict] = TOOLS,
    tool_registry: dict[str, tuple[Callable, list[str]]] = TOOL_REGISTRY,
    debug: bool = False
) -> str | AgentResult:
    """工具调用循环"""
    tool_debug = {
        "switch": debug,
        "tool_record": []
    }

    while True:
        message = await backend.chat_message(messages, tools = tools)

        if tool_calls:=message.tool_calls:
            messages.append(build_assistant_message(message))
            results = await tool_call_pipeline(tool_calls, tool_registry, tool_debug)

            for id, result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": json.dumps(result, ensure_ascii = False)
                })
        else:
            content = message.content or ""
            messages.append(build_assistant_message(message))

            if debug:
                return AgentResult(content, tool_debug["tool_record"])
            return content