"""
Node classification experiment for Graph Agentic Network
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import os
import json
import time
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import config
from gan.llm import LLMInterface, MockLLMInterface
from gan.graph import GraphAgenticNetwork
from gan.utils import seed_everything, evaluate_node_classification, compare_results
from data.dataset import load_or_create_dataset
from baselines.gcn import GCNBaseline
from baselines.gat import GAT
from baselines.graphsage import GraphSAGE

def run_node_classification(
    dataset_name: str = config.DATASET_NAME,
    use_subgraph: bool = True,
    subgraph_size: int = 1000,
    use_mock_llm: bool = False,
    num_layers: int = config.NUM_LAYERS,
    batch_size: Optional[int] = None,
    run_gan: bool = True,
    run_baselines: bool = True,
    save_results: bool = True,
    result_dir: Optional[str] = None,
    visualize: bool = True
) -> Dict[str, Any]:
    seed_everything()

    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(config.RESULTS_DIR, f"node_classification_{timestamp}")

    os.makedirs(result_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_name} (use_subgraph={use_subgraph}, size={subgraph_size})")
    dataset = load_or_create_dataset(dataset_name, use_subgraph, subgraph_size)

    adj_matrix = dataset['adj_matrix']
    node_features = dataset['node_features']
    labels = dataset['labels']
    with open("data/cora/cora_text_graph_simplified.jsonl") as f:
        node_texts = {
            int(json.loads(line)["node_id"]): json.loads(line)["text"]
            for line in f
        }
    train_idx = dataset['train_idx']
    val_idx = dataset['val_idx']
    test_idx = dataset['test_idx']
    num_classes = dataset['num_classes']

    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat()
    }

    if run_gan:
        config.LLM_BACKEND = "mock" if use_mock_llm else "remote"
        llm_interface = LLMInterface(model_name=config.LLM_MODEL)

        print(f"Creating Graph Agentic Network with {num_layers} layers")
        gan = GraphAgenticNetwork(
            adj_matrix=adj_matrix,
            llm_interface=llm_interface,
            labels=labels,
            node_texts=node_texts,
            num_layers=num_layers
        )

        print("Running Graph Agentic Network")
        start_time = time.time()
        gan.forward(batch_size=batch_size)
        gan_time = time.time() - start_time
        print(f"GAN completed in {gan_time:.2f} seconds")

        gan_predictions = gan.get_node_predictions()

        gan_metrics = {
            'all': evaluate_node_classification(gan_predictions, labels),
            'train': evaluate_node_classification(gan_predictions[train_idx], labels[train_idx]),
            'val': evaluate_node_classification(gan_predictions[val_idx], labels[val_idx]),
            'test': evaluate_node_classification(gan_predictions[test_idx], labels[test_idx])
        }

        print("GAN Results:")
        print(f"  Train Accuracy: {gan_metrics['train']['accuracy']:.4f}")
        print(f"  Val Accuracy: {gan_metrics['val']['accuracy']:.4f}")
        print(f"  Test Accuracy: {gan_metrics['test']['accuracy']:.4f}")

        results['gan'] = {
            'metrics': gan_metrics,
            'time': gan_time,
            'num_layers': num_layers,
            'llm_model': config.LLM_MODEL if not use_mock_llm else 'mock',
            'batch_size': batch_size
        }

    if run_baselines:
        edge_index = adj_matrix.nonzero().t().contiguous()
        baseline_results = {}

        for model_name in config.BASELINE_TYPES:
            print(f"\nRunning {model_name} baseline")
            if model_name == "GCN":
                from baselines.gcn import GCNBaseline as Baseline
                hidden_dim = config.GCN_HIDDEN_DIM
                dropout = config.GCN_DROPOUT
                lr = config.GCN_LEARNING_RATE
                weight_decay = config.GCN_WEIGHT_DECAY
                epochs = config.GCN_EPOCHS
            elif model_name == "GAT":
                hidden_dim = config.GAT_HIDDEN_DIM
                dropout = config.GAT_DROPOUT
                lr = config.GAT_LEARNING_RATE
                weight_decay = config.GAT_WEIGHT_DECAY
                epochs = config.GAT_EPOCHS
                class Baseline:
                    def __init__(self, in_channels, out_channels):
                        self.model = GAT(in_channels, hidden_dim, out_channels, dropout=dropout).to(config.DEVICE)
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
                        self.train_metrics = []
                        self.val_metrics = []
                    def train(self, edge_index, node_features, labels, train_idx, val_idx, epochs):
                        for epoch in range(epochs):
                            self.model.train()
                            self.optimizer.zero_grad()
                            out = self.model(node_features.to(config.DEVICE), edge_index.to(config.DEVICE))
                            loss = torch.nn.functional.cross_entropy(out[train_idx], labels[train_idx].to(config.DEVICE))
                            loss.backward()
                            self.optimizer.step()
                            acc = (out[train_idx].argmax(1) == labels[train_idx].to(config.DEVICE)).float().mean().item()
                            self.train_metrics.append({"loss": loss.item(), "acc": acc})
                    def evaluate(self, edge_index, node_features, labels, idx):
                        self.model.eval()
                        out = self.model(node_features.to(config.DEVICE), edge_index.to(config.DEVICE))
                        loss = torch.nn.functional.cross_entropy(out[idx], labels[idx].to(config.DEVICE))
                        pred = out[idx].argmax(1)
                        acc = (pred == labels[idx].to(config.DEVICE)).float().mean().item()
                        from sklearn.metrics import f1_score
                        f1_micro = f1_score(labels[idx].cpu(), pred.cpu(), average='micro')
                        f1_macro = f1_score(labels[idx].cpu(), pred.cpu(), average='macro')
                        return {"loss": loss.item(), "accuracy": acc, "f1_micro": f1_micro, "f1_macro": f1_macro}
            elif model_name == "GraphSAGE":
                hidden_dim = config.SAGE_HIDDEN_DIM
                dropout = config.SAGE_DROPOUT
                lr = config.SAGE_LEARNING_RATE
                weight_decay = config.SAGE_WEIGHT_DECAY
                epochs = config.SAGE_EPOCHS
                class Baseline:
                    def __init__(self, in_channels, out_channels):
                        self.model = GraphSAGE(in_channels, hidden_dim, out_channels, dropout=dropout).to(config.DEVICE)
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
                        self.train_metrics = []
                        self.val_metrics = []
                    def train(self, edge_index, node_features, labels, train_idx, val_idx, epochs):
                        for epoch in range(epochs):
                            self.model.train()
                            self.optimizer.zero_grad()
                            out = self.model(node_features.to(config.DEVICE), edge_index.to(config.DEVICE))
                            loss = torch.nn.functional.cross_entropy(out[train_idx], labels[train_idx].to(config.DEVICE))
                            loss.backward()
                            self.optimizer.step()
                            acc = (out[train_idx].argmax(1) == labels[train_idx].to(config.DEVICE)).float().mean().item()
                            self.train_metrics.append({"loss": loss.item(), "acc": acc})
                    def evaluate(self, edge_index, node_features, labels, idx):
                        self.model.eval()
                        out = self.model(node_features.to(config.DEVICE), edge_index.to(config.DEVICE))
                        loss = torch.nn.functional.cross_entropy(out[idx], labels[idx].to(config.DEVICE))
                        pred = out[idx].argmax(1)
                        acc = (pred == labels[idx].to(config.DEVICE)).float().mean().item()
                        from sklearn.metrics import f1_score
                        f1_micro = f1_score(labels[idx].cpu(), pred.cpu(), average='micro')
                        f1_macro = f1_score(labels[idx].cpu(), pred.cpu(), average='macro')
                        return {"loss": loss.item(), "accuracy": acc, "f1_micro": f1_micro, "f1_macro": f1_macro}

            model = Baseline(node_features.size(1), num_classes)
            start_time = time.time()
            model.train(edge_index, node_features, labels, train_idx, val_idx, epochs)
            elapsed = time.time() - start_time
            baseline_results[model_name.lower()] = {
                'metrics': {
                    'train': model.evaluate(edge_index, node_features, labels, train_idx),
                    'val': model.evaluate(edge_index, node_features, labels, val_idx),
                    'test': model.evaluate(edge_index, node_features, labels, test_idx),
                },
                'time': elapsed
            }
            print(f"{model_name} completed in {elapsed:.2f} seconds")

        results.update(baseline_results)

    if save_results:
        result_file = os.path.join(result_dir, "results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
        print(f"Results saved to {result_file}")

    return results

if __name__ == "__main__":
    run_node_classification(
        use_subgraph=False,
        subgraph_size=100,
        use_mock_llm=False,
        num_layers=config.NUM_LAYERS,
        batch_size=64,
        run_gan=False,
        run_baselines=True
    )
