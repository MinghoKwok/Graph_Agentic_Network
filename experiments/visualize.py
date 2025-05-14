"""
Visualization utilities for Graph Agentic Network experiments
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Any, Optional

import config
from gan.utils import visualize_node_embeddings, visualize_graph


def visualize_results(result_dir: str, output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = os.path.join(result_dir, "visualizations")

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "results.json")
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return

    with open(result_file, 'r') as f:
        results = json.load(f)

    _plot_accuracy_comparison(results, output_dir)
    _plot_time_comparison(results, output_dir)
    _plot_metric_breakdown(results, output_dir)

def _plot_accuracy_comparison(results: Dict[str, Any], output_dir: str):
    models = ['GAN'] + [k.upper() for k in results.keys() if k not in ['gan', 'dataset', 'timestamp']]
    train_acc = [results['gan']['metrics']['train']['accuracy']]
    val_acc = [results['gan']['metrics']['val']['accuracy']]
    test_acc = [results['gan']['metrics']['test']['accuracy']]

    for model in models[1:]:
        model_key = model.lower()
        train_acc.append(results[model_key]['metrics']['train']['accuracy'])
        val_acc.append(results[model_key]['metrics']['val']['accuracy'])
        test_acc.append(results[model_key]['metrics']['test']['accuracy'])

    x = np.arange(len(models))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, train_acc, width, label='Train')
    plt.bar(x, val_acc, width, label='Validation')
    plt.bar(x + width, test_acc, width, label='Test')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()

def _plot_time_comparison(results: Dict[str, Any], output_dir: str):
    models = ['GAN'] + [k.upper() for k in results.keys() if k not in ['gan', 'dataset', 'timestamp']]
    times = [results['gan']['time']]
    for model in models[1:]:
        model_key = model.lower()
        times.append(results[model_key]['time'])

    plt.figure(figsize=(8, 6))
    plt.bar(models, times)
    plt.ylabel('Time (seconds)')
    plt.title('Model Execution Time')
    for i, time_val in enumerate(times):
        plt.text(i, time_val + 0.5, f"{time_val:.2f}s", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()

def _plot_metric_breakdown(results: Dict[str, Any], output_dir: str):
    metrics = ['accuracy', 'f1_micro', 'f1_macro']
    for model in results:
        if model in ['dataset', 'timestamp']:
            continue
        model_metrics = results[model]['metrics']['test']
        values = [model_metrics[m] for m in metrics]
        plt.figure(figsize=(8, 6))
        plt.bar(metrics, values)
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title(f"{model.upper()} Test Set Metrics")
        for i, val in enumerate(values):
            plt.text(i, val + 0.02, f"{val:.4f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_metrics.png"))
        plt.close()

def visualize_node_decisions(gan, dataset, output_dir: str):
    node_memory = gan.get_node_memory()
    action_counts = {}
    for node_id, memory in node_memory.items():
        for entry in memory:
            action = entry['result']['action']
            action_counts[action] = action_counts.get(action, 0) + 1
    plt.figure(figsize=(10, 6))
    plt.bar(action_counts.keys(), action_counts.values())
    plt.title('Action Type Distribution')
    plt.xlabel('Action Type')
    plt.ylabel('Count')
    for i, (action, count) in enumerate(action_counts.items()):
        plt.text(i, count + 0.5, str(count), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_distribution.png"))
    plt.close()

    layers = max(len(m) for m in node_memory.values())
    layer_actions = {i: {} for i in range(layers)}
    for node_id, memory in node_memory.items():
        for entry in memory:
            layer = entry['layer']
            action = entry['result']['action']
            layer_actions[layer][action] = layer_actions[layer].get(action, 0) + 1

    fig, ax = plt.subplots(figsize=(12, 6))
    all_actions = sorted({a for d in layer_actions.values() for a in d})
    bottoms = np.zeros(layers)
    for action in all_actions:
        values = [layer_actions[i].get(action, 0) for i in range(layers)]
        ax.bar(range(layers), values, bottom=bottoms, label=action)
        bottoms += np.array(values)
    ax.set_title('Action Distribution by Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.set_xticks(range(layers))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_action_distribution.png"))
    plt.close()
