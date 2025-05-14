# data/cora/label_vocab.py

# 固定顺序的类别列表（来自 Cora 数据集）
# LABELS = [
#     "Case_Based",
#     "Genetic_Algorithms",
#     "Neural_Networks",
#     "Probabilistic_Methods",
#     "Reinforcement_Learning",
#     "Rule_Learning",
#     "Theory"
# ]

LABELS = [
    "Label_0",
    "Label_1",
    "Label_2",
    "Label_3",
    "Label_4",
    "Label_5",
    "Label_6"
]

# 映射：字符串 → 整数
label_vocab = {label: idx for idx, label in enumerate(LABELS)}

# 映射：整数 → 字符串
inv_label_vocab = {idx: label for label, idx in label_vocab.items()}

legacy_label_mapping = {
    "Case_Based": "Label_0",
    "Genetic_Algorithms": "Label_1",
    "Neural_Networks": "Label_2",
    "Probabilistic_Methods": "Label_3",   # 旧 -> 新
    "Reinforcement_Learning": "Label_4",
    "Rule_Learning": "Label_5",
    "Theory": "Label_6"                   # 旧 -> 新
}
