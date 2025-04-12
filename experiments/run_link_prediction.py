"""
Link prediction experiment for Graph Agentic Network
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import numpy as np
import config
from data.dataset import load_graph_data
from baselines.gcn import GCN
from baselines.gat import GAT
from baselines.graphsage import GraphSAGE
from gan.llm import LLMInterface
from gan.graph import GraphAgenticNetwork
from gan.utils import seed_everything, evaluate_link_prediction
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os
import matplotlib.pyplot as plt

def run_link_prediction(
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
        result_dir = os.path.join(config.RESULTS_DIR, f"link_prediction_{timestamp}")

    os.makedirs(result_dir, exist_ok=True)
    data = load_graph_data(dataset_name)

    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat()
    }

    if run_gan:
        print("\n[GAN] Running link prediction...")
        config.LLM_BACKEND = "mock" if use_mock_llm else "remote"
        llm_interface = LLMInterface(model_name=config.LLM_MODEL)

        gan = GraphAgenticNetwork(
            adj_matrix=data.adj_matrix,
            llm_interface=llm_interface,
            labels=data.y,
            node_texts=data.text,
            num_layers=num_layers
        )

        start_time = time.time()
        gan.forward(batch_size=batch_size)
        elapsed = time.time() - start_time
        predictions = gan.get_link_predictions()
        metrics = evaluate_link_prediction(predictions, data.y)
        results['gan'] = {
            'metrics': metrics,
            'time': elapsed,
            'num_layers': num_layers,
            'llm_model': config.LLM_MODEL if not use_mock_llm else 'mock',
            'batch_size': batch_size
        }
        print("[GAN] Completed.")

    if run_baselines:
        all_results = {}
        for model_name in config.BASELINE_TYPES:
            print(f"\n[Baseline: {model_name}] Running link prediction...")
            model_cls = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': GraphSAGE}[model_name]
            model = model_cls(
                in_channels=data.x.size(1),
                hidden_channels=256,
                out_channels=64
            ).to(config.DEVICE)

            edge_set = set([(u.item(), v.item()) for u, v in zip(*data.edge_index)])
            edge_list = list(edge_set)
            random.shuffle(edge_list)
            num_train = int(0.85 * len(edge_list))
            pos_train = edge_list[:num_train]
            pos_test = edge_list[num_train:]

            neg_edges = []
            while len(neg_edges) < len(edge_list):
                u, v = random.randint(0, data.num_nodes - 1), random.randint(0, data.num_nodes - 1)
                if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
                    neg_edges.append((u, v))
            neg_train = neg_edges[:num_train]
            neg_test = neg_edges[num_train:]

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            model.train()
            for _ in range(50):
                optimizer.zero_grad()
                z = model(data.x.to(config.DEVICE), data.edge_index.to(config.DEVICE))
                z = torch.nan_to_num(z, nan=0.0)
                loss = 0
                for (u, v) in pos_train:
                    loss += -torch.log(torch.sigmoid((z[u] * z[v]).sum()))
                for (u, v) in neg_train:
                    loss += -torch.log(1 - torch.sigmoid((z[u] * z[v]).sum()))
                loss.backward()
                optimizer.step()

            z = model(data.x.to(config.DEVICE), data.edge_index.to(config.DEVICE))
            z = torch.nan_to_num(z, nan=0.0)
            def score(pairs):
                return [torch.sigmoid((z[u] * z[v]).sum()).item() for u, v in pairs]
            y_true = [1] * len(pos_test) + [0] * len(neg_test)
            y_score = score(pos_test + neg_test)

            all_results[model_name.lower()] = {
                'metrics': evaluate_link_prediction(y_true, y_score),
                'time': None  # optional: record if needed
            }

        results.update(all_results)

    if save_results:
        with open(os.path.join(result_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_link_prediction(
        run_gan=True,
        run_baselines=True
    )
