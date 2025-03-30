# Graph Agentic Network vs GNN Baseline Comparison

## ✅ GAN vs GNN：节点分类任务对照图

| 维度 | GNN（GCN/GAT等） | GAN（Graph Agentic Network） |
|------|------------------|------------------------------|
| **学习范式** | 半监督（semi-supervised） | 自主智能（agent-based） |
| **标签使用方式** | 训练集标签用于监督梯度更新 | 可选（mockLLM可用 label，真实 LLM 可零标签） |
| **邻居特征处理** | 明确 message passing，固定函数（如 aggregate） | 智能 agent 决定何时拉取、广播、更新 |
| **无邻居节点** | 只能靠自身特征，精度低 | LLM 有机会泛化典型节点模式（待扩展） |
| **训练方式** | 端到端神经网络反向传播 | 每层 forward，由 LLM 控制行为，无梯度 |
| **输出** | 节点的类别（或 embedding） | 节点预测标签、内部状态、行为轨迹等 |
| **实验评价指标** | accuracy@train/val/test | 同上；另可评估 agent 决策合理性 |
| **可解释性** | 弱，可做 attention 可视化 | 强，可跟踪每一步行为 + prompt |

---

## 🧪 Baseline 实验准备清单

### 1. 数据准备

- [x] 使用 OGBN-Arxiv 或类似数据集
- [x] 保留原始划分：`train_idx` / `val_idx` / `test_idx`
- [x] 可选择 `subgraph_size` 用于 GAN 快速实验（e.g. 1000）

### 2. GCN 训练基线

- [x] 在 `train_idx` 上训练
- [x] 在 `val_idx` 调参（早停、模型选择）
- [x] 在 `test_idx` 上报告最终结果
- [x] 输出指标：train/val/test accuracy，loss 曲线，可视化（已集成）

### 3. GAN 智能图模型

- [x] 支持 mockLLM / real LLM 切换
- [x] 每层 agent 自主决策
- [x] 当前支持预测标签、拉取邻居、广播消息、更新状态
- [x] 每层 `step_summary` 输出（你正在开发）
- [x] 支持 `debug_single_node.py` 跟踪单个 agent

### 4. GAN vs. GCN 实验对齐要求

| 要求 | 是否满足 | 备注 |
|------|----------|------|
| 相同数据集与划分 | ✅ | 用 `ogbn-arxiv` 同一组索引 |
| 评估指标一致 | ✅ | accuracy 为主，可扩展 F1 等 |
| 输出结果记录 | ✅ | `results.json` 中保存 |
| 可视化对比 | ✅ | bar plot、training curve 已集成 |
| 单节点调试工具 | ✅ | `scripts/debug_single_node.py` |
| 多节点信息传递可视化 | ⏳ | 正在开发 Graphviz 支持 |
| LLM prompt 跟踪 | ✅ | 现已打印每层每节点 prompt |

---

## 🧠 你下一步可能考虑：

- [ ] 消息流可视化（Graphviz）
- [ ] Agent 行为摘要图（每层 update/retrieve/broadcast）
- [ ] 引导无邻居节点自学（典型节点引导机制设计）
- [ ] 不同数据类型扩展（TAG → image-graph 等）
- [ ] Few-shot 标签自动引导机制（仍保持 fully-auto 本质）
