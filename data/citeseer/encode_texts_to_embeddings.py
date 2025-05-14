# encode_texts_to_embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm

INPUT_PATH = "citeseer_text_graph_simplified.jsonl"
OUTPUT_PATH = "citeseer_text_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    # 加载模型
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Loaded sentence-transformer: {MODEL_NAME}")

    # 加载文本数据
    records = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            node_id = int(record["node_id"])  # 👈 强制转换为 int
            records.append((node_id, record["text"]))

    node_ids = [nid for nid, _ in records]
    texts = [text for _, text in records]

    # 批量编码
    print(f"⚙️ Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # 为 cosine 检索准备
    )

    # 存为 {node_id (int): embedding} dict
    result = {nid: emb for nid, emb in zip(node_ids, embeddings)}
    np.save(OUTPUT_PATH, result)
    print(f"✅ Saved embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
