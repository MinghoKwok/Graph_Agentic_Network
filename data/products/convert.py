import pandas as pd
import json
import numpy as np
from tqdm import tqdm

# === File paths ===
data_path = "data.csv"             # paper_id, label
node_info_path = "node_info_clean.csv"   # paper_id, title, abstract
graph_path = "graph.csv"           # paper_id1, paper_id2

out_jsonl = "products_text_graph.jsonl"
out_edge = "products.cites"
out_embeddings = "products_text_embeddings.npy"

# === Load data ===
data_df = pd.read_csv(data_path)
node_info_df = pd.read_csv(node_info_path)
data_df = pd.merge(data_df, node_info_df, on="paper_id", how="left")

graph_df = pd.read_csv(graph_path)

# === Build paper_id <-> node_id mapping ===
unique_paper_ids = data_df["paper_id"].unique()
paper_id_to_node_id = {pid: idx for idx, pid in enumerate(unique_paper_ids)}

# === Generate JSONL with human-readable labels ===
with open(out_jsonl, "w") as f_out:
    for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
        node_id = paper_id_to_node_id[row["paper_id"]]
        label = str(row["label"]).strip()

        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        text = f"{title}\n{abstract}"

        entry = {
            "node_id": node_id,
            "paper_id": int(row["paper_id"]),
            "text": text,
            "label": label  # 🔁 保留原始 label 字符串
        }
        f_out.write(json.dumps(entry) + "\n")

# === Generate edge list ===
with open(out_edge, "w") as f_out:
    for _, row in tqdm(graph_df.iterrows(), total=len(graph_df)):
        src = paper_id_to_node_id.get(row["paper_id1"])
        tgt = paper_id_to_node_id.get(row["paper_id2"])
        if src is not None and tgt is not None:
            f_out.write(f"{src}\t{tgt}\n")

# === Placeholder embeddings ===
N = len(paper_id_to_node_id)
D = 768
embeddings = np.random.randn(N, D).astype(np.float32)
np.save(out_embeddings, embeddings)

print(f"✅ Files written:\n- {out_jsonl}\n- {out_edge}\n- {out_embeddings}")
