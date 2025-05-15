# data/citeseer/label_vocab.py

# 固定顺序的类别列表（来自 Cora 数据集）
# LABELS = [
# Agents
# ML
# IR
# DB
# HCI
# AI
# ]

LABELS = [
    "Label_0",
    "Label_1",
    "Label_2",
    "Label_3",
    "Label_4",
    "Label_5"
]

# 映射：字符串 → 整数
label_vocab = {label: idx for idx, label in enumerate(LABELS)}

# 映射：整数 → 字符串
inv_label_vocab = {idx: label for label, idx in label_vocab.items()}

legacy_label_mapping = {
    "Agents": "Label_0",
    "ML": "Label_1",
    "IR": "Label_2",
    "DB": "Label_3",
    "HCI": "Label_4",
    "AI": "Label_5"
}

# legacy_label_mapping = {
#     "Agents": "Agents",
#     "ML": "Machine Learning",
#     "IR": "Information Retrieval",
#     "DB": "Databases",
#     "HCI": "Human-Computer Interaction",
#     "AI": "Artificial Intelligence"
# }

# Agents,Agents
# ML,Machine Learning
# IR,Information Retrieval
# DB,Databases
# HCI,Human-Computer Interaction
# AI,Artificial Intelligence
