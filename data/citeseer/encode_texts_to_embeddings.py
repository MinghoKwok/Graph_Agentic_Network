from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm

INPUT_PATH = "citeseer_text_graph_simplified.jsonl"
OUTPUT_PATH = "citeseer_text_embeddings_msmarco-MiniLM-L6-cos-v5.npy"  # 👈 建议不同模型不同文件名
MODEL_NAME = "msmarco-MiniLM-L6-cos-v5"  # 👈 可切换为你要尝试的模型

# 🔁 模型选择建议：
# - "BAAI/bge-base-en-v1.5"       : 🔥推荐！针对 dense retrieval 优化，语义表达强，适合学术类文本
# - "intfloat/e5-base-v2"         : 支持指令式语义嵌入（Instruct Embedding），效果也很稳
# - "msmarco-MiniLM-L6-cos-v5"    : 微调于检索任务，比 all-MiniLM 更适合向量搜索
# - "all-MiniLM-L6-v2"            : 原版通用模型（当前用的），速度快但检索能力较弱

def main():
    # 加载模型
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Loaded embedding model: {MODEL_NAME}")

    # 加载文本数据
    records = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            node_id = int(record["node_id"])
            text = record["text"].strip()
            if not text:
                continue  # 👈 跳过空文本
            records.append((node_id, text))

    node_ids = [nid for nid, _ in records]
    texts = [text for _, text in records]

    print(f"⚙️ Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    result = {nid: emb for nid, emb in zip(node_ids, embeddings)}
    np.save(OUTPUT_PATH, result)
    print(f"✅ Saved {len(result)} embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
