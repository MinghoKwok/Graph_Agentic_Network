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
    "Label_6",
    "Label_7",
    "Label_8",
    "Label_9",
    "Label_10",
    "Label_11",
    "Label_12",
    "Label_13",
    "Label_14",
    "Label_15",
    "Label_16",
    "Label_17",
    "Label_18",
    "Label_19",
    "Label_20",
    "Label_21",
    "Label_22",
    "Label_23",
    "Label_24",
    "Label_25",
    "Label_26",
    "Label_27",
    "Label_28",
    "Label_29",
    "Label_30",
    "Label_31",
    "Label_32",
    "Label_33",
    "Label_34",
    "Label_35",
    "Label_36",
    "Label_37",
    "Label_38",
    "Label_39"
]

# 映射：字符串 → 整数
label_vocab = {label: idx for idx, label in enumerate(LABELS)}

# 映射：整数 → 字符串
inv_label_vocab = {idx: label for label, idx in label_vocab.items()}

legacy_label_mapping = {
    "Numerical Analysis": "Label_0",
    "Multimedia": "Label_1",
    "Logic in Computer Science": "Label_2",
    "Computers and Society": "Label_3",
    "Cryptography and Security": "Label_4",
    "Distributed and Parallel and Cluster Computing": "Label_5",
    "Human-Computer Interaction": "Label_6",
    "Computational Engineering and Finance and Science": "Label_7",
    "Networking and Internet Architecture": "Label_8",
    "Computational Complexity": "Label_9",
    "Artificial Intelligence": "Label_10",
    "Multiagent Systems": "Label_11",
    "General Literature": "Label_12",
    "Neural and Evolutionary Computing": "Label_13",    
    "Symbolic Computation": "Label_14",
    "Hardware Architecture": "Label_15",
    "Computer Vision and Pattern Recognition": "Label_16",
    "Graphics": "Label_17",
    "Emerging Technologies": "Label_18",
    "Systems and Control": "Label_19",
    "Computational Geometry": "Label_20",
    "Other Computer Science": "Label_21",
    "Programming Languages": "Label_22",
    "Software Engineering": "Label_23",
    "Machine Learning": "Label_24",
    "Sound": "Label_25",    
    "Social and Information Networks": "Label_26",      
    "Robotics": "Label_27",
    "Information Theory": "Label_28",
    "Performance": "Label_29",
    "Computation and Language": "Label_30",
    "Information Retrieval": "Label_31",    
    "Mathematical Software": "Label_32",    
    "Formal Languages and Automata Theory": "Label_33",
    "Data Structures and Algorithms": "Label_34",
    "Operating Systems": "Label_35",
    "Computer Science and Game Theory": "Label_36",
    "Databases": "Label_37",
    "Digital Libraries": "Label_38",
    "Discrete Mathematics": "Label_39"
}
