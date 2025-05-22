import torch
import json
import numpy as np

# 加载文件（替换为你本地路径）
features = torch.load("features.pt")
labels = torch.load("labels.pt")
edge_index = torch.load("edge_index.pt")

# 构造 label vocab
unique_labels = sorted(labels.unique().tolist())
inv_label_vocab = {int(v): f"Label_{int(v)}" for v in unique_labels}

# 保存 .cites
with open("chameleon.cites", "w") as f:
    for src, dst in edge_index.t().tolist():
        f.write(f"{src} {dst}\n")

# 保存 .jsonl（text_graph）
top_k = 5
with open("chameleon_text_graph_simplified.jsonl", "w") as f:
    for i in range(features.size(0)):
        topk = torch.topk(features[i], top_k).indices.tolist()
        keywords = [f"word_{j}" for j in topk]
        text = f"This article discusses {', '.join(keywords)}."
        entry = {
            "node_id": int(i),
            "paper_id": int(i),
            "text": text,
            "label": inv_label_vocab[int(labels[i])]
        }
        f.write(json.dumps(entry) + "\n")

# 保存 .npy（可选）
embeddings = {i: features[i].numpy() for i in range(features.size(0))}
np.save("chameleon_text_embeddings.npy", embeddings)
