"""
Node classification experiment for Graph Agentic Network
"""

import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time
import argparse
import torch
from torch_geometric.datasets import OGB_MAG
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import os
import json
import pandas as pd
from datetime import datetime

from baselines.gcn import GCNBaseline
import config
from gan.utils import evaluate_node_classification as evaluate_accuracy, load_dataset, log_node_predictions
from gan.llm import RemoteLLMInterface, MockLLMInterface
from experiments.visualize import visualize_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GAN', choices=['GAN', 'GCN'], help='Model to run')
    parser.add_argument('--use_llm', action='store_true', help='Use real LLM via vLLM')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset name')
    parser.add_argument('--log_path', type=str, default='logs/', help='Path to save logs')
    parser.add_argument('--visualize', action='store_true', help='Enable message trace visualization')
    args = parser.parse_args()

    # Set config
    config.USE_LOCAL_LLM = args.use_llm
    llm_interface = RemoteLLMInterface() if args.use_llm else MockLLMInterface()

    # Load dataset
    data, num_classes = load_dataset(args.dataset)

    if args.model == 'GCN':
        model = GCN(in_channels=data.x.size(-1), hidden_channels=256, out_channels=num_classes)
    elif args.model == 'GAN':
        model = GraphAgentNetwork(
            in_channels=data.x.size(-1),
            hidden_channels=256,
            out_channels=num_classes,
            llm_interface=llm_interface,
            visualize=args.visualize
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = model.to(config.DEVICE)
    data = data.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_acc = 0.0

    log_dir = os.path.join(args.log_path, f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    summary_records = []

    start_time = time.time()

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = out.argmax(dim=1)
        acc = evaluate_accuracy(pred, data)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(loss),
            "val_acc": float(acc['val']),
            "test_acc": float(acc['test']),
            "time_elapsed": time.time() - start_time
        }

        if args.model == 'GAN' and hasattr(model, 'step_summaries'):
            step_path = os.path.join(log_dir, f"step_summary_epoch{epoch:03d}.json")
            with open(step_path, 'w') as f:
                json.dump(model.step_summaries, f, indent=2)

        summary_records.append(epoch_summary)

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {acc['val']:.4f} | Test Acc: {acc['test']:.4f}")

        if acc['val'] > best_acc:
            best_acc = acc['val']
            log_node_predictions(pred, data.y, data, log_dir, model_type=args.model)

    elapsed = time.time() - start_time
    print("Training finished.")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Total Time: {elapsed:.2f} seconds")

    # Save summary CSV
    df = pd.DataFrame(summary_records)
    df.to_csv(os.path.join(log_dir, "training_summary.csv"), index=False)

    # Generate results.json for visualization
    metrics = {
        "train": {
            "accuracy": float(acc['train'])
        },
        "val": {
            "accuracy": float(acc['val'])
        },
        "test": {
            "accuracy": float(acc['test']),
            "f1_micro": float(acc.get('f1_micro', 0)),
            "f1_macro": float(acc.get('f1_macro', 0))
        }
    }
    results_json_path = os.path.join(log_dir, "results.json")
    results_data = {
        args.model.lower(): {
            "metrics": metrics,
            "time": round(elapsed, 2)
        }
    }
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Auto visualize
    visualize_results(log_dir)


if __name__ == '__main__':
    main()


# python experiments/node_classification.py --model GAN --use_llm --dataset ogbn-arxiv --log_path logs/ --visualize
