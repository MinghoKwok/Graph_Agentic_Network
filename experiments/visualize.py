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
    """
    Visualize experiment results.
    
    Args:
        result_dir: Directory containing experiment results
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = os.path.join(result_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    result_file = os.path.join(result_dir, "results.json")
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Plot accuracy comparison
    _plot_accuracy_comparison(results, output_dir)
    
    # Plot time comparison
    _plot_time_comparison(results, output_dir)
    
    # Plot metric breakdown
    _plot_metric_breakdown(results, output_dir)


def _plot_accuracy_comparison(results: Dict[str, Any], output_dir: str):
    """
    Plot accuracy comparison between models.
    
    Args:
        results: Experiment results
        output_dir: Output directory
    """
    # Extract metrics
    gan_train = results['gan']['metrics']['train']['accuracy']
    gan_val = results['gan']['metrics']['val']['accuracy']
    gan_test = results['gan']['metrics']['test']['accuracy']
    
    gcn_train = results['gcn']['metrics']['train']['accuracy']
    gcn_val = results['gcn']['metrics']['val']['accuracy']
    gcn_test = results['gcn']['metrics']['test']['accuracy']
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    models = ['GAN', 'GCN']
    train_acc = [gan_train, gcn_train]
    val_acc = [gan_val, gcn_val]
    test_acc = [gan_test, gcn_test]
    
    x = np.arange(len(models))
    width = 0.25
    
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
    """
    Plot time comparison between models.
    
    Args:
        results: Experiment results
        output_dir: Output directory
    """
    # Extract times
    gan_time = results['gan']['time']
    gcn_time = results['gcn']['time']
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    models = ['GAN', 'GCN']
    times = [gan_time, gcn_time]
    
    plt.bar(models, times)
    plt.ylabel('Time (seconds)')
    plt.title('Model Execution Time')
    
    # Add time labels on top of bars
    for i, time in enumerate(times):
        plt.text(i, time + 1, f"{time:.2f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()


def _plot_metric_breakdown(results: Dict[str, Any], output_dir: str):
    """
    Plot metric breakdown for each model.
    
    Args:
        results: Experiment results
        output_dir: Output directory
    """
    # First for GAN
    gan_metrics = results['gan']['metrics']['test']
    metrics = ['accuracy', 'f1_micro', 'f1_macro']
    values = [gan_metrics[m] for m in metrics]
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('GAN Test Set Metrics')
    
    for i, val in enumerate(values):
        plt.text(i, val + 0.02, f"{val:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gan_metrics.png"))
    plt.close()
    
    # Then for GCN
    gcn_metrics = results['gcn']['metrics']['test']
    values = [gcn_metrics[m] for m in metrics]
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('GCN Test Set Metrics')
    
    for i, val in enumerate(values):
        plt.text(i, val + 0.02, f"{val:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gcn_metrics.png"))
    plt.close()


def visualize_node_decisions(gan, dataset, output_dir: str):
    """
    Visualize node decision patterns.
    
    Args:
        gan: Trained GAN model
        dataset: Dataset dictionary
        output_dir: Output directory
    """
    # Extract node memory
    node_memory = gan.get_node_memory()
    
    # Count actions by type
    action_counts = {}
    for node_id, memory in node_memory.items():
        for entry in memory:
            action = entry['result']['action']
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
    
    # Plot action distribution
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
    
    # Plot action distribution by layer
    layers = max(len(m) for m in node_memory.values())
    layer_actions = {i: {} for i in range(layers)}
    
    for node_id, memory in node_memory.items():
        for entry in memory:
            layer = entry['layer']
            action = entry['result']['action']
            
            if action not in layer_actions[layer]:
                layer_actions[layer][action] = 0
            layer_actions[layer][action] += 1
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all unique actions
    all_actions = set()
    for layer_dict in layer_actions.values():
        all_actions.update(layer_dict.keys())
    all_actions = sorted(all_actions)
    
    # Prepare data
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
