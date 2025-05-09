from torch_geometric.datasets import WikipediaNetwork
import torch
import json

# 设置数据集的根目录
root = './data'

# 加载 Chameleon 数据集
dataset = WikipediaNetwork(root=root, name='chameleon', geom_gcn_preprocess=True)

# 获取数据对象
data = dataset[0]

# 打印数据集的信息
print(data)

# 保存核心数据
torch.save(data.x, 'data/chameleon/features.pt')
torch.save(data.y, 'data/chameleon/labels.pt')
torch.save(data.edge_index, 'data/chameleon/edge_index.pt')

# 保存划分索引
torch.save(data.train_mask, 'data/chameleon/train_mask.pt')
torch.save(data.val_mask, 'data/chameleon/val_mask.pt')
torch.save(data.test_mask, 'data/chameleon/test_mask.pt')



# construct feature vocab

vocab = {str(i): f"word_{i}" for i in range(2325)}  # 2325 是 feature 维度
with open('data/chameleon/feature_vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)



# feature to TAG


features = torch.load('data/chameleon/features.pt')
vocab = json.load(open('data/chameleon/feature_vocab.json'))

text_list = []
for row in features:
    indices = row.nonzero(as_tuple=True)[0].tolist()
    words = [vocab[str(i)] for i in indices]
    text = "This node is about: " + ", ".join(words)
    text_list.append(text)

with open('data/chameleon/raw_texts.json', 'w') as f:
    json.dump(text_list, f, indent=2)

