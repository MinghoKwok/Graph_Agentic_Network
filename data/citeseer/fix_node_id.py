import json

input_path = "citeseer_text_graph_simplified.jsonl"
output_path = "citeseer_text_graph_simplified.jsonl"

# 读取所有节点
with open(input_path, "r") as f:
    nodes = [json.loads(line) for line in f]

# 按顺序加上 node_id 字段
for idx, node in enumerate(nodes):
    node["node_id"] = idx

# 覆盖写回文件
with open(output_path, "w") as f:
    for node in nodes:
        f.write(json.dumps(node, ensure_ascii=False) + "\n")

print("已为每个节点加上 node_id 字段，并从0递增。")
