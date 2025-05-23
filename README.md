# Graph Agentic Network (GAN)

**Node is Agent!**  
A novel graph learning paradigm where each node acts as an autonomous agent, powered by a frozen Large Language Model (LLM). This replaces traditional GNN message passing with agentic, self-decided behavior: retrieve, update, broadcast, or no-op.

---

## 🚀 Features

- **LLM-powered agents** — Nodes use a frozen LLM to make local decisions.
- **Global semantic retrieval** — Nodes access far-away context via RAG.
- **Per-node planning** — Each node independently chooses its action per layer.
- **Support for node classification** — Works on datasets like Cora, Citeseer, Pubmed.
- **Debuggable & interpretable** — Memory, prompts, responses, and actions are all logged.

---

## 📦 Installation

```bash
git clone https://github.com/MinghoKwok/Graph_Agentic_Network.git
cd Graph_Agentic_Network
pip install -r requirements.txt
```

---

## 🔧 Quick Start

### Run node classification

```bash
python experiments/run_node_classification.py
```

### Run with specific GPU

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_node_classification.py
```

### Debug a single node

```bash
python scripts/debug_single_node.py --node_id 100 --layer 0 --subgraph_size 10000
```

---

## ⚡ Run LLM Inference Server (vLLM)

```bash
conda activate vllm_env
python3 -m vllm.entrypoints.openai.api_server \
  --model ./models/llama-3.1-8b-instruct \
  --tokenizer ./models/llama-3.1-8b-instruct \
  --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.95
```

---

## 📁 Project Structure

```text
gan/                # Core logic for agents, actions, graph, and LLM
  ├── node.py       # NodeAgent class
  ├── actions.py    # Action definitions
  ├── graph.py      # AgenticGraph class
  ├── llm.py        # RemoteLLM and fallback logic
  └── utils.py      # Prompt builders, deduplication, etc.
experiments/        # Run scripts and visualization tools
data/               # Dataset and label vocab
config.py           # Central config file
```

---

## ⚙️ Configuration

Edit `config.py` to configure:

- Dataset (Cora / Citeseer / Pubmed)
- Number of layers
- LLM mode (`mock` or `remote`)
- Prompt strategy and retrieval options

---

## 📚 Citation

```bibtex
@misc{graph-agentic-network,
  author = {Minghao Guo},
  title = {Graph Agentic Network: LLM-powered Decision Making for Graph Learning},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/MinghoKwok/Graph_Agentic_Network}}
}
```

---

## 🪪 License

MIT License
