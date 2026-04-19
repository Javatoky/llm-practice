import asyncio

from Phase_preparation.async_stream_chat import chat
from .Chromadb_practice import DB_NAME, METADATA, client
from .embedding_practice import get_embeddings


def prompt_modification(filtered_results: list) -> str:
    context_parts = []
    for i, (doc, meta) in enumerate(filtered_results, 1):
        source = meta.get("source")
        context_parts.append(f"{i}. 来源：{source}\n{doc}")
    context = "\n\n---\n\n".join(context_parts)
    system_prompt = f"""
        你是一个问答助手。请只基于参考资料回答用户的问题。
        如果参考资料中没有相关信息，请明确说"根据现有资料无法回答此问题"。
        不要编造任何不在参考资料中的信息。
        回答务必包含来源标注
        ## 参考资料：
        {context}
    """.strip()
    return system_prompt

async def rag_query(question: str, top_k: int = 3, distance_threshold: float = 0.7):
    # 向量化提问内容
    question_vector  = await get_embeddings([question])

    # 检索Top-K
    collection = client.get_or_create_collection(
        name = DB_NAME,
        metadata = METADATA
    )
    results = collection.query(
        query_embeddings = question_vector,
        n_results = top_k
    )
    # 按距离阈值过滤
    documents = results.get("documents")[0]
    metadatas = results.get("metadatas")[0]
    distances = results.get("distances")[0]
    filtered_results = [
        (doc, meta) for doc, meta, dist
        in zip(documents, metadatas, distances)
        if dist < distance_threshold
    ]
    if not filtered_results:
        print("根据现有资料无法回答此问题")
        return

    # 拼接进系统提示词
    system_prompt = prompt_modification(filtered_results)

    # 调用大模型回答（流式输出）
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    async for token in chat(messages, question):
        print(token, end ="", flush = True)

test_cases = [
    "Gemini提示词中的迭代优化设计是怎样的",
    "我领悟到的好的交友关系应该是怎样的",
    "平板究竟该怎么用才能让自己的学习效率提高",
]

async def main():
    print("测试开始：\n")
    for i, case in enumerate(test_cases, 1):
        print(f"---Test{i}---")
        await rag_query(case)
        print("\n\n")
    print("测试结束")

if __name__ == "__main__":
    asyncio.run(main())

    

