"""
Quick demo script for Graph Agentic Network with minimal setup
"""

import os
import sys
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from gan.llm import MockLLMInterface
from gan.graph import GraphAgenticNetwork


def generate_synthetic_graph(num_nodes=10, feature_dim=5, num_classes=3):
    """Generate a small synthetic graph for demo purposes."""
    # Create a random graph
    G = nx.fast_gnp_random_graph(num_nodes, 0.3)
    
    # Ensure the graph is connected
    while not nx.is_connected(G):
        G = nx.fast_gnp_random_graph(num_nodes, 0.3)
    
    # Create adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    for u, v in G.edges():
        adj_matrix[u, v] = 1.0
        adj_matrix[v, u] = 1.0  # Undirected graph
    
    # Create node features (random)
    node_features = torch.randn((num_nodes, feature_dim))
    
    # Create node labels (random)
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    return adj_matrix, node_features, labels, G


def main():
    print("Running Graph Agentic Network Quick Demo")
    
    # Generate a small synthetic graph
    print("Generating synthetic graph...")
    num_nodes = 10
    feature_dim = 5
    num_classes = 3
    adj_matrix, node_features, labels, G = generate_synthetic_graph(num_nodes, feature_dim, num_classes)
    
    # Visualize the original graph
    print("Visualizing original graph...")
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    plt.title("Synthetic Graph Structure")
    plt.savefig("synthetic_graph.png")
    plt.close()
    
    # Initialize mock LLM (for quick demo)
    print("Initializing MockLLM...")
    llm = MockLLMInterface()
    
    # Create GAN
    print("Creating Graph Agentic Network...")
    gan = GraphAgenticNetwork(
        adj_matrix=adj_matrix,
        node_features=node_features,
        llm_interface=llm,
        labels=labels,
        num_layers=2
    )
    
    # Run GAN
    print("Running Graph Agentic Network...")
    gan.forward()
    
    # Get predictions
    print("Getting predictions...")
    predictions = gan.get_node_predictions()
    
    # Print results
    print("\nResults:")
    print(f"{'Node':<5} {'True Label':<15} {'Predicted Label':<15}")
    print("-" * 35)
    for i in range(num_nodes):
        true_label = labels[i].item()
        pred_label = predictions[i].item() if i < len(predictions) else "N/A"
        print(f"{i:<5} {true_label:<15} {pred_label:<15}")
    
    # Calculate accuracy
    correct = (predictions == labels).sum().item()
    accuracy = correct / num_nodes
    print(f"\nAccuracy: {accuracy:.2f} ({correct}/{num_nodes})")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
