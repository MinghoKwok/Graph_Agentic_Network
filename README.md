# Graph Agentic Network (GAN)

Node is Agent!

A novel approach to graph learning where each node functions as an autonomous agent, powered by large language models for decision-making. This framework replaces traditional message passing in graph neural networks with agent-based decision making.

## Features

- 🧠 **LLM-powered node agents**: Each node uses an LLM to make decisions based on its current state
- 🔄 **Flexible communication patterns**: Nodes can choose which neighbors to interact with
- 🤔 **Autonomous decision-making**: Nodes decide when to retrieve information, broadcast messages, or update their state
- 📊 **Support for graph ML tasks**: Node classification, link prediction, and more

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/graph-agentic-network.git
cd graph-agentic-network
```

2. Set up the environment:
```bash
bash setup.sh
conda activate gan
```

3. Download the datasets:
```bash
bash download_data.sh
```

## Quick Start

Run a node classification experiment on the OGB-Arxiv dataset:

```bash
python experiments/run_node_classification.py
```
```bash
CUDA_VISIBLE_DEVICES=4 python experiments/run_node_classification.py
```


```bash
python scripts/debug_single_node.py --node_id 100 --layer 0 --subgraph_size 10000
```


## vLLM
```bash
conda activate vllm_env
python3 -m vllm.entrypoints.openai.api_server \
  --model /common/home/mg1998/Graph/GAN/Graph_Agentic_Network/models/llama-3.1-8b-instruct \
  --tokenizer /common/home/mg1998/Graph/GAN/Graph_Agentic_Network/models/llama-3.1-8b-instruct \
  --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.95 \
  --served-model-name llama-3.1-8b-instruct
```

```bash
(vllm_qwen3) mg1998@wisehub:/common/users/mg1998/models$ /common/users/mg1998/envs/vllm_qwen3/bin/python -m vllm.entrypoints.openai.api_server   --model /common/users/mg1998/models/Qwen3-14B   --tokenizer /common/users/mg1998/models/Qwen3-14B   --port 8001

```

## Project Structure

- `gan/`: Core framework components
  - `actions.py`: Action classes for node agents
  - `node.py`: Node agent implementation
  - `graph.py`: Graph and network implementation
  - `llm.py`: LLM interface
- `data/`: Data handling utilities
- `baselines/`: Baseline implementations (e.g., GCN)
- `experiments/`: Experiment scripts

## Configuration

Edit `config.py` to adjust experiment parameters and model settings.

## Citation

If you use this code in your research, please cite our work:
```
@misc{graph-agentic-network,
  author = {Your Name},
  title = {Graph Agentic Network: LLM-powered Decision Making for Graph Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/graph-agentic-network}}
}
```

## License

MIT License