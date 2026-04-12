import asyncio

import chromadb
from Document_chunk_loop import process_document, test_file_paths
from embedding_practice import get_embeddings

test_cases = [
    "Gemini 3提示词中给Gemini设定的Goal",
    "真正的友谊",
    "Trump最喜欢的食物"
]

client = chromadb.PersistentClient(path = "./chromadb")
DB_NAME = "my_documents"
METADATA = {"hnsw:space": "cosine"}

async def chromadb_storage():
    """向量数据库存储"""
    # 文档处理管线
    processed_documents = [process_document(filepath) for filepath in test_file_paths]      # 思考：这里是一次性读取，内存压力很大，或许我们应该实现一个惰性读取的方法
    
    # embedding向量化
    embedding_results = [get_embeddings([chunk.get("text") for chunk in chunks]) for chunks in processed_documents]
    embedding_results = await asyncio.gather(*embedding_results)

    # 存入向量数据库
    client.delete_collection("my_documents")
    collection = client.get_or_create_collection(
        name = DB_NAME,
        metadata = METADATA
    )
    for doc, doc_embeddings in zip(processed_documents, embedding_results):
        collection.upsert(     # 思考：这里存入文档也是一次性全部存入，但如果数据大了一次性就会出现性能问题，而且存入也属于io流，这是否可以用异步来节省时间呢？
            ids = [f"{chunk.get('metadata').get('source')}_{chunk.get('metadata').get('chunk_index')}" for chunk in doc],
            documents = [chunk.get("text") for chunk in doc],
            embeddings = doc_embeddings,
            metadatas =[chunk.get("metadata") for chunk in doc]
        )
    
async def main():
    """查询测试"""
    collection = client.get_or_create_collection(
        name = DB_NAME,
        metadata = METADATA
    )

    case_embeddings = await get_embeddings(test_cases)
    results = collection.query(
        query_embeddings = case_embeddings,
        n_results = 3
    )
    documents = results.get("documents")
    metadatas = results.get("metadatas")
    distances = results.get("distances")

    for i, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start = 1):
        print(f"\n---Test{i}---")
        for j, (doc, md, dis) in enumerate(zip(document, metadata, distance), start = 1):
            print(f"文本{j}: ", end = '')
            print(doc[:50])
            print(f"元数据：{md}")
            print(f"距离：{dis}")

if __name__ == "__main__":
    asyncio.run(main())