from pathlib import Path


def file_read(file_path: str) -> str:
    """读取文本"""
    with open(Path(file_path), "r", encoding = "utf-8") as f:
        text = f.read()
    return text

def chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
    """分档处理"""
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    # 这里处理的逻辑是每次拆分文章时新建新的字符串将拆分的两端分别赋值给新字符串，先按段落分块
    chunks = []
    while True:
        if not (text:=text.strip()):
            break
        # 这里文本强制要求以\n\n分段，所以这里先用\n\n分段
        para, _, text = text.partition('\n\n')      # TODO:增强鲁棒性，匹配任意空格分隔的段落
        if not para.strip():
            continue
        if (length:=len(para)) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while True:
                if (end:=(start + chunk_size)) > length:
                    chunks.append(para[start: length])
                    break 
                chunks.append(para[start: end])
                start = end - overlap  
    return chunks

def metadata_add(file_path: str, chunks: list[str]) -> list[dict]:
    """元数据处理"""
    final_chunks = []
    for i, chunk in enumerate(chunks):
        final_chunks.append(
            {
                "text": chunk,
                "metadata": {
                    "source": Path(file_path).name,
                    "chunk_index": i
                }
            }
        )
    return final_chunks

def process_document(file_path: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """文档处理管线"""
    text = file_read(file_path)
    chunks = chunking(text, chunk_size, overlap)
    return metadata_add(file_path, chunks)

test_file_paths = [
    "./test_documents/gemini_prompt.md",
    "./test_documents/my_view_on_friendship.md"
]

def main():
    print("测试开始\n")
    for i, file_path in enumerate(test_file_paths, 1):
        print(f"---Test{i}---\n")
        document_chunks = process_document(file_path)
        for j, chunk in enumerate(document_chunks):
            print(f"- chunk_{j}: ")
            content = chunk.get("text")
            metadata = chunk.get("metadata")
            print(f"content: {content}")
            print(f"metadata: {metadata}")
        print("\n")
    print("测试结束")

if __name__ == "__main__":
    main()

