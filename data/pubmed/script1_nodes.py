import json

# 三个部分
paths = [
    "sampled_2_10_train.jsonl",
    "sampled_2_10_val.jsonl",
    "sampled_2_10_test.jsonl"
]

output_path = "pubmed_text_graph_simplified.jsonl"

with open(output_path, "w", encoding="utf-8") as fout:
    for path in paths:
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line)
                node_id = obj["id"]
                text = obj["text"]
                fout.write(json.dumps({
                    "node_id": node_id,
                    "paper_id": node_id,
                    "text": text
                }, ensure_ascii=False) + "\n")

print("✅ pubmed_text_graph_simplified.jsonl 构建完成")
