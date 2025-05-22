# 固定顺序的类别列表（Chameleon 中通常是整数 0~5）
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

# 如果未来你想补充语义别名映射，可以放这里
legacy_label_mapping = {
    "Label_0": "Label_0",
    "Label_1": "Label_1",
    "Label_2": "Label_2",
    "Label_3": "Label_3",
    "Label_4": "Label_4",
    "Label_5": "Label_5",                   
}
