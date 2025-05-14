# Converts Citeseer/Arxiv-style CSVs to GAN-compatible files:
# - arxiv_text_graph_simplified.jsonl
# - arxiv.cites
# - arxiv_text_embeddings.npy (optional placeholder)

import pandas as pd
import json
import numpy as np
from tqdm import tqdm

# === File paths (you can change these as needed) ===
data_path = "data.csv"             # Contains: paper_id, label, title, abstract
graph_path = "graph.csv"          # Contains: paper_id1, paper_id2
label_info_path = "label_info.csv" # Contains: label idx, arxiv category

out_jsonl = "arxiv_text_graph_simplified.jsonl"
out_edge = "arxiv.cites"
out_embeddings = "arxiv_text_embeddings.npy"  # Placeholder only

# === Load files ===
# Load data.csv (contains paper_id + label)
data_df = pd.read_csv(data_path)
# Load node_info.csv (contains paper_id + title + abstract)
node_info_df = pd.read_csv("node_info.csv")  # 👈 文件名可能需要改
# Merge them on paper_id
data_df = pd.merge(data_df, node_info_df, on="paper_id", how="left")

graph_df = pd.read_csv(graph_path)
label_info_df = pd.read_csv(label_info_path, names=["label idx", "arxiv category"], header=0, usecols=[0, 1])

# === Build paper_id <-> node_id mapping ===
unique_paper_ids = data_df["paper_id"].unique()
paper_id_to_node_id = {pid: idx for idx, pid in enumerate(unique_paper_ids)}

# === Build label mapping (optional: human-readable labels) ===
label_id_to_name = dict(zip(label_info_df["label idx"], label_info_df["arxiv category"]))

# === Generate JSONL: node_id, paper_id, text (title+abstract), label_name ===
with open(out_jsonl, "w") as f_out:
    for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
        node_id = paper_id_to_node_id[row["paper_id"]]
        label_idx = int(row["label"])
        label_name = label_id_to_name.get(label_idx, f"Unknown_Label_{label_idx}")

        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        text = f"{title}\n{abstract}"

        entry = {
            "node_id": node_id,
            "paper_id": int(row["paper_id"]),
            "text": text,
            "label": label_name
        }
        f_out.write(json.dumps(entry) + "\n")

# === Generate .cites edge list ===
with open(out_edge, "w") as f_out:
    for _, row in tqdm(graph_df.iterrows(), total=len(graph_df)):
        src = paper_id_to_node_id.get(row["paper_id1"])
        tgt = paper_id_to_node_id.get(row["paper_id2"])
        if src is not None and tgt is not None:
            f_out.write(f"{src}\t{tgt}\n")

# === Placeholder for embeddings (random vectors) ===
N = len(paper_id_to_node_id)
D = 768
embeddings = np.random.randn(N, D).astype(np.float32)
np.save(out_embeddings, embeddings)

print(f"✅ Files written:\n- {out_jsonl}\n- {out_edge}\n- {out_embeddings} (random placeholder)")
