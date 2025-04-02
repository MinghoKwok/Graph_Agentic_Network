"""
Data loading and processing utilities for Graph Agentic Network
"""

import torch
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx
import json
import config
from gan.utils import seed_everything
from data.cora.label_vocab import label_vocab


def load_ogb_arxiv(root: str = os.path.join(config.DATA_DIR, 'ogbn-arxiv')) -> Dict[str, Any]:
    """
    Load the OGB-Arxiv dataset.
    
    Args:
        root: Directory to store the dataset
        
    Returns:
        Dictionary containing dataset components
    """
    print(f"Loading OGB-Arxiv dataset from {root}...")
    seed_everything()
    
    # Load dataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    # Extract components
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Create adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj_matrix[edge_index[0], edge_index[1]] = 1.0
    
    # Get features and labels
    node_features = data.x
    labels = data.y.squeeze()
    
    # Get train/val/test splits
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    print(f"Dataset loaded: {num_nodes} nodes, {edge_index.size(1)} edges")
    print(f"Feature dimension: {node_features.size(1)}, Number of classes: {labels.max().item() + 1}")
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    return {
        'adj_matrix': adj_matrix,
        'edge_index': edge_index,
        'node_features': node_features,
        'labels': labels,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'num_classes': labels.max().item() + 1,
        'num_nodes': num_nodes
    }

def load_cora(jsonl_path: str = "data/cora/cora_text_graph_simplified.jsonl") -> Dict[str, Any]:
    """
    Load Cora dataset from a preprocessed JSONL file with text and labels.
    Assumes file format: {"node_id": int, "text": str, "label": int}
    """
    print(f"Loading Cora text dataset from {jsonl_path}")
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    node_texts = {}
    labels = {}
    for line in lines:
        item = json.loads(line)
        node_id = int(item["node_id"])
        node_texts[node_id] = item["text"]
        labels[node_id] = item["label"]

    num_nodes = len(node_texts)
    labels_tensor = torch.tensor([label_vocab[labels[i]] for i in range(num_nodes)], dtype=torch.long)


    # Dummy adjacency: fully connected graph or empty — replace if you have real edges
    adj_matrix = torch.eye(num_nodes)

    # Train/val/test split (e.g., 60/20/20)
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    return {
        'adj_matrix': adj_matrix,
        'node_features': torch.zeros((num_nodes, 1)),  # Dummy for GCN baseline
        'labels': labels_tensor,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'num_classes': labels_tensor.max().item() + 1,
        'num_nodes': num_nodes
    }


def create_subgraph(adj_matrix: torch.Tensor, node_features: torch.Tensor, 
                   labels: torch.Tensor, subset_size: int = 1000) -> Dict[str, Any]:
    """
    Create a subgraph for testing or experimentation.
    
    Args:
        adj_matrix: Full adjacency matrix
        node_features: Full feature matrix
        labels: Full label tensor
        subset_size: Size of the subgraph to create
        
    Returns:
        Dictionary containing subgraph components
    """
    num_nodes = adj_matrix.size(0)
    
    # Random node selection
    seed_everything()
    chosen_indices = torch.randperm(num_nodes)[:subset_size]
    
    # Create subgraph adjacency matrix
    sub_adj = adj_matrix[chosen_indices][:, chosen_indices]
    
    # Extract relevant features and labels
    sub_features = node_features[chosen_indices]
    sub_labels = labels[chosen_indices]
    
    # Create train/val/test split
    perm = torch.randperm(subset_size)
    train_size = int(subset_size * 0.6)
    val_size = int(subset_size * 0.2)
    
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size+val_size]
    test_idx = perm[train_size+val_size:]
    
    print(f"Created subgraph with {subset_size} nodes and {sub_adj.sum().item()/2:.0f} edges")
    
    return {
        'adj_matrix': sub_adj,
        'node_features': sub_features,
        'labels': sub_labels,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'num_classes': labels.max().item() + 1,
        'num_nodes': subset_size,
        'original_indices': chosen_indices
    }


def convert_to_pytorch_geometric(adj_matrix: torch.Tensor, node_features: torch.Tensor, 
                                labels: torch.Tensor) -> Tuple:
    """
    Convert data to PyTorch Geometric format.
    
    Args:
        adj_matrix: Adjacency matrix
        node_features: Node feature matrix
        labels: Node labels
        
    Returns:
        Tuple of (edge_index, node_features, labels)
    """
    # Convert adjacency matrix to edge_index
    edge_index = adj_matrix.nonzero().t().contiguous()
    
    return edge_index, node_features, labels


def load_or_create_dataset(name: str = config.DATASET_NAME, 
                           use_subgraph: bool = False, 
                           subgraph_size: int = 1000) -> Dict[str, Any]:
    """
    Load a dataset or create a subgraph from it.
    
    Args:
        name: Dataset name
        use_subgraph: Whether to create a subgraph
        subgraph_size: Size of the subgraph
        
    Returns:
        Dictionary containing dataset components
    """
    if name == 'ogbn-arxiv':
        dataset = load_ogb_arxiv()
    elif name == 'cora':
        dataset = load_cora()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    if use_subgraph:
        return create_subgraph(
            dataset['adj_matrix'], 
            dataset['node_features'], 
            dataset['labels'], 
            subgraph_size
        )
    
    return dataset
