import asyncio

import numpy as np
from openai import AsyncOpenAI

from config import QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL

client = AsyncOpenAI(
    api_key = QWEN_ALY_API_KEY,
    base_url = QWEN_ALY_BASE_URL
)

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(
        model = "text-embedding-v1",
        input = texts
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))    # 向量的余弦计算公式

test_cases = [
    "今天天气不错，万里无云",
    "天气真好，艳阳天",
    "Python编程是大模型应用的基石",
    "日式料理称得上是一个美食",
    "晴朗的天气能让人心情愉悦"
]

async def main():
    print('相似度矩阵: ')
    vectors = await get_embeddings(test_cases)
    print(f"{'vectors':>4}", end = "")
    for i in range(1, len(vectors)+1):
        print(f"{i:^8}", end = "")
    print()
    for i, row_vector in enumerate(vectors, start = 1):
        print(f"{i:>4}", end = '')
        for column_vector in vectors:
            print(f"{cosine_similarity(row_vector, column_vector):>8.3f}", end = "")
        print()

if __name__ == "__main__":
    asyncio.run(main())
    
            

