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

def load_cora(
    jsonl_path: str = "data/cora/cora_text_graph_simplified.jsonl",
    edge_path: str = "data/cora/cora.cites"
) -> Dict[str, Any]:
    """
    Load Cora dataset using simplified JSONL + true citation edges.
    
    Returns:
        Dictionary containing:
        - adj_matrix: Adjacency matrix for graph structure
        - node_features: One-hot node features for GNN
        - node_texts: Node text descriptions for GAN
        - labels: Node labels
        - train_idx, val_idx, test_idx: Dataset splits
        - num_classes: Number of label classes
        - num_nodes: Total number of nodes
    """
    print(f"🔄 Loading Cora dataset from {jsonl_path} and {edge_path}")

    # Step 1: 加载节点信息
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    node_texts = {}
    paperid_to_nodeid = {}
    labels = {}

    for line in lines:
        item = json.loads(line)
        nid = int(item["node_id"])
        pid = int(item["paper_id"])
        node_texts[nid] = item["text"]
        legacy_label_mapping = {
            "Case_Based": "Case_Based",
            "Genetic_Algorithms": "Genetic_Algorithms",
            "Neural_Networks": "Neural_Networks",
            "Probabilistic_Methods": "Probabilistic_Methods",   # 旧 -> 新
            "Reinforcement_Learning": "Reinforcement_Learning",
            "Rule_Learning": "Rule_Learning",
            "Theory": "Label_6"                   # 旧 -> 新
        }
        # 然后在处理 label 时加：
        label_name = legacy_label_mapping.get(item["label"], item["label"])  # 先映射一遍

        labels[nid] = label_vocab[label_name]
        paperid_to_nodeid[pid] = nid

    num_nodes = len(node_texts)
    labels_tensor = torch.tensor([labels[i] for i in range(num_nodes)], dtype=torch.long)

    # Step 2: 构建邻接矩阵（真实边）
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    with open(edge_path, "r") as f:
        for line in f:
            src_pid, tgt_pid = map(int, line.strip().split())
            if src_pid in paperid_to_nodeid and tgt_pid in paperid_to_nodeid:
                src = paperid_to_nodeid[src_pid]
                tgt = paperid_to_nodeid[tgt_pid]
                adj_matrix[src, tgt] = 1
                adj_matrix[tgt, src] = 1  # 无向边

    # Step 3: 创建 GNN 输入特征（one-hot）
    node_features = torch.eye(num_nodes)

    # Step 4: 划分 train/val/test
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    print(f"Loaded {num_nodes} nodes with {adj_matrix.sum().item()/2:.0f} edges")
    print(f"Feature dimension: {node_features.size(1)}")
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return {
        'adj_matrix': adj_matrix,
        'node_features': node_features,  # GNN 输入
        'node_texts': node_texts,        # GAN 输入
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
        Dictionary containing:
        - adj_matrix: Subgraph adjacency matrix
        - node_features: Subgraph node features
        - labels: Subgraph labels
        - train_idx, val_idx, test_idx: Subgraph splits
        - num_classes: Number of classes
        - num_nodes: Number of nodes in subgraph
        - original_indices: Original node indices in full graph
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
    print(f"Feature dimension: {sub_features.size(1)}")
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    return {
        'adj_matrix': sub_adj,
        'node_features': sub_features,
        'labels': sub_labels,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'num_classes': labels.max().item() + 1,
        'num_nodes': subset_size,
        'original_indices': chosen_indices  # 用于映射回原始图的 node_texts
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
        name: Dataset name ('cora' or 'ogbn-arxiv')
        use_subgraph: Whether to create a subgraph
        subgraph_size: Size of the subgraph
        
    Returns:
        Dictionary containing:
        - adj_matrix: Adjacency matrix
        - node_features: Node features for GNN
        - node_texts: Node texts for GAN
        - labels: Node labels
        - train_idx, val_idx, test_idx: Dataset splits
        - num_classes: Number of classes
        - num_nodes: Number of nodes
    """
    if name == 'ogbn-arxiv':
        dataset = load_ogb_arxiv()
        # OGB-Arxiv 已经有 node_features，但需要添加 node_texts
        dataset['node_texts'] = {i: f"Paper {i}" for i in range(dataset['num_nodes'])}
    elif name == 'cora':
        dataset = load_cora()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    if use_subgraph:
        # 创建子图时保持 node_texts 的映射
        subgraph = create_subgraph(
            dataset['adj_matrix'], 
            dataset['node_features'], 
            dataset['labels'], 
            subgraph_size
        )
        # 更新 node_texts 映射
        original_indices = subgraph['original_indices']
        subgraph['node_texts'] = {
            new_idx: dataset['node_texts'][old_idx]
            for new_idx, old_idx in enumerate(original_indices)
        }
        return subgraph
    
    return dataset

def load_graph_data(dataset_name: str):
    """
    Return PyG-compatible graph data (edge_index, x, etc.) for link prediction tasks.
    """
    import torch
    from torch_geometric.data import Data

    if dataset_name == "cora":
        dataset = load_cora()
    elif dataset_name == "ogbn-arxiv":
        dataset = load_ogb_arxiv()
    else:
        raise ValueError(f"Unsupported dataset for link prediction: {dataset_name}")

    # Extract edge_index from adjacency matrix
    edge_index = dataset['adj_matrix'].nonzero().t().contiguous()

    # Build PyG Data object
    data = Data(
        edge_index=edge_index,
        x=dataset['node_features'],
        y=dataset['labels']
    )
    data.num_nodes = dataset['num_nodes']

    return data
