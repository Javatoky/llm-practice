from pathlib import Path

from openai import AsyncOpenAI

import chromadb
from config import (DS_API_KEY, DS_BASE_URL, DS_MODEL, QWEN_ALY_API_KEY,
                    QWEN_ALY_BASE_URL)


def readfile(filepath: str) -> str:
    """读取文本"""
    path = Path(filepath)
    with open(path, "r", encoding = "utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError("The text is empty")
    return text

def get_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """分块处理"""
    if chunk_size <= overlap:
        raise ValueError("overlap must be less than chunk_size")
    chunks: list[str] = []
    while True:
        if not (text:=text.strip()):
            break
        para, _, text = text.strip().partition("\n\n")
        if not para.strip():
            continue
        if (length:=len(para))<chunk_size:
            chunks.append(para)
        else:
            start = 0
            while True:
                if (end:=start + chunk_size) > length:
                    chunks.append(para[start: length])
                    break
                chunks.append(para[start: end])
                start = end - overlap
    return chunks

def metadata_add(filepath: str, chunks: list[str]) -> list[dict]:
    """元数据处理"""
    processed_chunks: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        chunks.append(
            {
                "text": chunk,
                "metadata": {
                    "source": Path(filepath).name(),
                    "chunk_index": i
                }
            }
        )
    return processed_chunks

embedding_client = AsyncOpenAI(
    api_key = QWEN_ALY_API_KEY,
    base_uel = QWEN_ALY_BASE_URL,
)
async def get_embeddings(texts: list[str]):
    """向量化文本"""
    result = await embedding_client.embeddings.create(
        input = [text for text in texts],
        model = "text-embedding-v1"
    )

    return [item.embedding for item in result.data]

chromadb_client = chromadb.PersistentClient(path = "./chromadb")
def chromadb_storage(chunks: list[dict], chunk_embeddings: list[list[float]]):
    """存入Chromadb数据库"""
    collection = chromadb_client.get_or_create_collection(
        name = "my_document",
        metadata = {"hnsw:space": "cosine"}
    )
    collection.add(
        ids = [chunk["chunk_index"] for chunk in chunks],
        embeddings = chunk_embeddings,
        documents = [chunk["text"] for chunk in chunks],
        metadatas = [chunk["metadata"] for chunk in chunks]
    )

async def doc_process_pipeline(filepath: str):
    try:
        text = readfile(filepath)
        chunks = get_chunks(text)
    except ValueError:
        raise
    processed_chunks = metadata_add(chunks)
    embeddings = await get_embeddings(chunks)
    chromadb_storage(processed_chunks, embeddings)

llm_client = AsyncOpenAI(
    api_key = DS_API_KEY,
    base_url = DS_BASE_URL
)
async def chat(user_input: str, messages: list[dict]):
    messages.append({"role": "user", "content": user_input})
    response = await llm_client.chat.completions.create(
        model = DS_MODEL,
        messages = messages,
        stream = True
    )
    reply = ""
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            reply += delta
            yield delta
    messages.append({"role": "assistant", "content": reply})

async def QA_pipeline(query: str, top_k: int = 3, distance_threshold: float = 0.7):
    embedding = await get_embeddings([query])
    collection = chromadb_client.get_or_create_collection(
        name = "my_document",
        metadata = {"hnsw:space": "cosine"}
    )
    results = collection.query(
        query_embeddings = embedding,
        n_results = top_k
    )
    documents = results.get("documents")[0]
    metadatas = results.get("metadatas")[0]
    distances = results.get("distances")[0]
    filtered_results = [
        (doc, meta) for doc, meta, dict
        in zip(documents, metadatas, distances)
        if dict < distance_threshold
    ] 
    system_prompt = """
    ## Role: 你是一个问答助手，能根据用户的问题输入，严格根据参考文本回答
    ## Constrains & Capability: 
        - 用户输入中会包含**参考文本**和**问题**两部分，你只能根据**参考文本**回答用户的**问题**
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
        助手：根据参考资料，元认知的提升对个人成长有如下作用：
        1. 打破惯性反应：在刺激与行动间创造“暂停键”，变无意识反应为主动选择。
        2. 升级学习模式：不只纠错答案，更溯源思维漏洞，将失败转化为底层算法的迭代。
        3. 消解情绪内耗：抽离出“观察者视角”，在痛苦中剥离意义，提升心理韧性。
        4. 实现跨域迁移：从“学会知识”进阶到“学会如何学习”，让成长方式本身不断进化。
        【来源】metacognitive_role.md
    """.strip()

