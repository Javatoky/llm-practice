import copy
import json
from pathlib import Path

from model_adapter import LLMBackend

import chromadb

chromadb_client = chromadb.PersistentClient(path = str(Path(__file__).parent / "chromadb"))

RAG_CONFIG = {
    "top_k": 3,
    "distance_threshold": 0.7
}

def get_chromadb_client(path_str: str = "../chromadb"):
    return chromadb.PersistentClient(path = str)

def readfile(filepath: str) -> str:
    """读取文本"""
    path = Path(filepath)
    with open(path, "r", encoding = "utf-8") as f:
        text = f.read()
    return text

def get_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """分块处理"""
    if chunk_size <= overlap:
        raise ValueError("参数overlap必须小于chunk_size！")
    chunks: list[str] = []
    while True:
        if not (text:=text.strip()):
            break
        para, _, text = text.partition("\n\n")
        if not para.strip():
            continue
        if (length:=len(para))<chunk_size:
            chunks.append(para)
        else:
            start = 0
            while True:
                if (end:=start + chunk_size) >= length:
                    chunks.append(para[start: length])
                    break
                chunks.append(para[start: end])
                start = end - overlap
    return chunks

def add_metadata(filepath: str, chunks: list[str]) -> list[dict]:
    """元数据处理"""
    processed_chunks: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        processed_chunks.append(
            {
                "text": chunk,
                "metadata": {
                    "source": Path(filepath).name,
                    "chunk_index": i
                }
            }
        )
    return processed_chunks

def store_into_chromadb(collection: chromadb.Collection, chunks_with_metadatas: list[dict], chunk_embeddings: list[list[float]]) -> None:
    """存入Chromadb数据库"""
    collection.upsert(
        ids = [f"{chunk.get('metadata').get('source')}_{chunk.get('metadata').get('chunk_index')}" for chunk in chunks_with_metadatas],
        embeddings = chunk_embeddings,
        documents = [chunk["text"] for chunk in chunks_with_metadatas],
        metadatas = [chunk["metadata"] for chunk in chunks_with_metadatas]
    )

async def build_vector_store(backend: LLMBackend, filepaths: list[str], collection_name: str) -> None:
    """文档存储流"""
    collection = chromadb_client.get_or_create_collection(
        name = collection_name,
        metadata = {"hnsw:space": "cosine"}
    )

    for filepath in filepaths:
        text = readfile(filepath)
        chunks = get_chunks(text)
        processed_chunks = add_metadata(filepath, chunks)
        embeddings = await backend.embed(chunks)
        store_into_chromadb(collection, processed_chunks, embeddings)

async def retrieve(backend: LLMBackend, query: str, collection_name: str, top_k: int = RAG_CONFIG["top_k"], distance_threshold: float = RAG_CONFIG["distance_threshold"]) -> list[dict]:
    """信息检索"""
    try:
        collection = chromadb_client.get_collection(
            name = collection_name,
        )
    except ValueError as e:
        raise ValueError(f"Collection not found: {collection_name}")

    embedding = await backend.embed([query])
    results = collection.query(
        query_embeddings = embedding,
        n_results = top_k
    )

    documents = results.get("documents")[0]
    metadatas = results.get("metadatas")[0]
    distances = results.get("distances")[0]
    filtered_results = [
        (doc, meta, dist) for doc, meta, dist
        in zip(documents, metadatas, distances)
        if dist < distance_threshold
    ]
    if not filtered_results:
        return []
    
    texts: list[dict] = []
    for doc, meta, dist in filtered_results:
        texts.append({
            "reference_text": f"{doc}",
            "source": f"{meta.get('source')}",
            "chunk_index": f"{meta.get('chunk_index')}",
            "distance": f"{dist}"
        })
        #context_parts.append(f"{i}. 来源：{source}\n{doc}")
    #context = "\n\n---\n\n".join(context_parts)
    return texts

async def answer_with_rag(
    backend: LLMBackend,
    user_input: str,
    collection_name: str,
    messages: list[dict]
) -> str:
    """问答过程"""
    texts = await retrieve(backend, user_input, collection_name)
    if not texts:
        reply = f"根据现有资料无法回答此问题"
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": reply})
        return reply
    
    temp_messages = copy.deepcopy(messages)
    temp_messages.append(
        {
            "role": "user", 
            "content":
                f"【参考文本信息】\n"
                f"{json.dumps(texts, ensure_ascii=False)}\n\n"
                f"【用户问题】{user_input}"
        }
    )
    reply = await backend.chat(temp_messages)

    messages.append(
        {"role": "user", "content": user_input}
    )
    messages.append(
        {"role": "assistant", "content": reply}
    )

    return reply