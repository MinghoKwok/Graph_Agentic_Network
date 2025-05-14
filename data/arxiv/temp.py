import json

# 原始文件路径（label 是 int）
input_path = "arxiv_text_graph_simplified.jsonl"

# 输出文件路径（label 是 str）
output_path = "arxiv_text_graph_simplified_cora_style.jsonl"

# idx → 类别名称映射
idx_to_category = {
    0: "Numerical Analysis",
    1: "Multimedia",
    2: "Logic in Computer Science",
    3: "Computers and Society",
    4: "Cryptography and Security",
    5: "Distributed and Parallel and Cluster Computing",
    6: "Human-Computer Interaction",
    7: "Computational Engineering and Finance and Science",
    8: "Networking and Internet Architecture",
    9: "Computational Complexity",
    10: "Artificial Intelligence",
    11: "Multiagent Systems",
    12: "General Literature",
    13: "Neural and Evolutionary Computing",
    14: "Symbolic Computation",
    15: "Hardware Architecture",
    16: "Computer Vision and Pattern Recognition",
    17: "Graphics",
    18: "Emerging Technologies",
    19: "Systems and Control",
    20: "Computational Geometry",
    21: "Other Computer Science",
    22: "Programming Languages",
    23: "Software Engineering",
    24: "Machine Learning",
    25: "Sound",
    26: "Social and Information Networks",
    27: "Robotics",
    28: "Information Theory",
    29: "Performance",
    30: "Computation and Language",
    31: "Information Retrieval",
    32: "Mathematical Software",
    33: "Formal Languages and Automata Theory",
    34: "Data Structures and Algorithms",
    35: "Operating Systems",
    36: "Computer Science and Game Theory",
    37: "Databases",
    38: "Digital Libraries",
    39: "Discrete Mathematics"
}

def convert_labels(input_path: str, output_path: str):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            item = json.loads(line)
            label_idx = item["label"]
            label_str = idx_to_category.get(label_idx, f"Unknown_Label_{label_idx}")
            item["label"] = label_str
            outfile.write(json.dumps(item) + "\n")

    print(f"✅ Converted label idx → label name: {output_path}")

if __name__ == "__main__":
    convert_labels(input_path, output_path)
