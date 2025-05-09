import json
import random

# ✅ 加载 feature_vocab（确保你已更新为真实语义词表）
with open("feature_vocab.json") as f:
    feature_vocab = json.load(f)

# ✅ 获取所有可选词（去重 + 排序可选）
vocab_list = list(set(feature_vocab.values()))
vocab_list.sort()

# ✅ 设置参数
num_nodes = 2277  # 请根据你的实际节点数量调整
min_words, max_words = 5, 30  # 每个节点文本中的词数范围

# ✅ 为每个节点随机选词并构造文本
output = []
for nid in range(num_nodes):
    k = random.randint(min_words, max_words)
    words = random.sample(vocab_list, k)
    text = f"This node is about: {', '.join(words)}"
    output.append(text)

# ✅ 保存为 raw_texts.json
with open("raw_texts.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Successfully generated raw_texts.json for {num_nodes} nodes.")
