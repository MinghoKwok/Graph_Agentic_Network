import pandas as pd
import json

# 读取数据
node_info = pd.read_csv('data/citeseer/node_info.csv')
labels = pd.read_csv('data/citeseer/data.csv')

# 创建标签映射
unique_labels = sorted(labels['label'].unique())
label_vocab = {label: f"label_{i}" for i, label in enumerate(unique_labels)}

# 合并数据
merged_data = pd.merge(node_info, labels, on='paper_id', how='left')

# 创建JSONL文件
with open('data/citeseer/citeseer_text_graph_simplified.jsonl', 'w') as f:
    for _, row in merged_data.iterrows():
        # 组合title和abstract
        text = f"{row['title']} {row['abs']}"
        
        # 创建节点数据
        node_data = {
            "node_id": str(row['paper_id']),
            "text": text,
            "label": label_vocab[row['label']]  # 使用映射后的标签
        }
        
        # 写入JSONL文件
        f.write(json.dumps(node_data) + '\n')

# 保存标签映射
with open('data/citeseer/label_vocab.py', 'w') as f:
    f.write('label_vocab = {\n')
    for label, mapped_label in label_vocab.items():
        f.write(f"    '{label}': '{mapped_label}',\n")
    f.write('}\n\n')
    
    f.write('inv_label_vocab = {\n')
    for label, mapped_label in label_vocab.items():
        f.write(f"    '{mapped_label}': '{label}',\n")
    f.write('}\n') 