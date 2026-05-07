import asyncio
import sys
from pathlib import Path

from model_adapter import LLMBackend, ModelConfig
from rag_agent import answer_with_rag, build_vector_store

sys.path.insert(0, str(Path(__file__).parent.parent))


from config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                    QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)

test_filepaths = [
    "../test_documents/gemini_prompt.md",
    "../test_documents/my_view_on_friendship.md",
    "../test_documents/ai_and_coding.md",
    "../test_documents/metacognitive_role.md",
    "../test_documents/personal_prioritization_under_multiple_tasks.md"
]
# 转换为相对于当前脚本所在目录的绝对路径
script_dir = Path(__file__).parent
test_filepaths = [str(script_dir / path) for path in test_filepaths]

test_cases = [
    "Gemini 3提示词中给Gemini设定的Goal",
    "真正的友谊",
    "Trump最喜欢的食物"
]

system_prompt = """
    ## Role: 你是一个问答助手，能根据用户的问题输入，严格根据参考文本回答
    ## Constrains & Capability: 
        - 用户输入中会包含**参考文本（包含来源）**和**问题**两部分，你只能根据**参考文本**回答用户的**问题**
        - 如果**参考文本**中没有相关内容或没有内容必须回答“根据现有资料无法回答此问题”
        - 不可编造任何参考资料中没有的内容
        - 回答必须包含参考资料的来源标注
    ## Knowledge: 用户输入中的所有**参考文本**的内容
    ## Output Format: 
        - 尽可能结构化输出
        - 内容简要、直击要点
    ## Example Output: 
        问答示例：
        用户：元认知的提升对一个人成长的作用是怎样的?
        助手：根据参考资料metacognitive_role.md 的 chunk_1、chunk_2，元认知的提升对个人成长有如下作用：
        1. 打破惯性反应：在刺激与行动间创造“暂停键”，变无意识反应为主动选择。
        2. 升级学习模式：不只纠错答案，更溯源思维漏洞，将失败转化为底层算法的迭代。
        3. 消解情绪内耗：抽离出“观察者视角”，在痛苦中剥离意义，提升心理韧性。
        4. 实现跨域迁移：从“学会知识”进阶到“学会如何学习”，让成长方式本身不断进化。
    """.strip()

async def main():
    """主流程"""
    backend = LLMBackend(ModelConfig(
        chat_model = QWEN_MODEL,
        chat_api_key = QWEN_API_KEY,
        chat_base_url = QWEN_BASE_URL,
        embed_model = QWEN_EMBED_MODEL,
        embed_api_key = QWEN_ALY_API_KEY,
        embed_base_url = QWEN_ALY_BASE_URL,
        supports_embeddings = True
    ))

    await build_vector_store(backend, test_filepaths, "my_documents")
    for i, case in enumerate(test_cases, 1):
        print(f"\n---Test{i}---")
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        print(f"用户：{case}")
        messages.append(
            {"role": "user", "content": case}
        )
        reply = await answer_with_rag(backend, case, "my_documents", messages)
        print(f"助手：{reply}")

if __name__ == "__main__":
    asyncio.run(main())