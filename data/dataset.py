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
from data.chameleon.label_vocab import label_vocab, inv_label_vocab
from torch_geometric.data import Data

def load_ogb_arxiv(root: str = os.path.join(config.DATA_DIR, 'ogbn-arxiv')) -> Dict[str, Any]:
    print(f"Loading OGB-Arxiv dataset from {root}...")
    seed_everything()
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj_matrix[edge_index[0], edge_index[1]] = 1.0
    node_features = data.x
    labels = data.y.squeeze()
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

def load_cora(jsonl_path: str = "/common/home/mg1998/Graph/GAN/Graph_Agentic_Network/data/cora/cora_text_graph_simplified.jsonl",
               edge_path: str = "/common/home/mg1998/Graph/GAN/Graph_Agentic_Network/data/cora/cora.cites") -> Dict[str, Any]:
    print(f"🔄 Loading Cora dataset from {jsonl_path} and {edge_path}")
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
            "Case_Based": "Label_0",
            "Genetic_Algorithms": "Label_1",
            "Neural_Networks": "Label_2",
            "Probabilistic_Methods": "Label_3",
            "Reinforcement_Learning": "Label_4",
            "Rule_Learning": "Label_5",
            "Theory": "Label_6"
        }
        label_name = legacy_label_mapping.get(item["label"], item["label"])
        labels[nid] = label_vocab[label_name]
        paperid_to_nodeid[pid] = nid
    num_nodes = len(node_texts)
    labels_tensor = torch.tensor([labels[i] for i in range(num_nodes)], dtype=torch.long)
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    with open(edge_path, "r") as f:
        for line in f:
            src_pid, tgt_pid = map(int, line.strip().split())
            if src_pid in paperid_to_nodeid and tgt_pid in paperid_to_nodeid:
                src = paperid_to_nodeid[src_pid]
                tgt = paperid_to_nodeid[tgt_pid]
                adj_matrix[src, tgt] = 1
                adj_matrix[tgt, src] = 1
    node_features = torch.eye(num_nodes)
    perm = torch.randperm(num_nodes)
    train_size = int(0.79 * num_nodes)
    val_size = int(0.01 * num_nodes)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    print(f"Loaded {num_nodes} nodes with {adj_matrix.sum().item()/2:.0f} edges")
    print(f"Feature dimension: {node_features.size(1)}")
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return {
        'adj_matrix': adj_matrix,
        'node_features': node_features,
        'node_texts': node_texts,
        'labels': labels_tensor,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'num_classes': labels_tensor.max().item() + 1,
        'num_nodes': num_nodes
    }

def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def load_or_create_dataset(name: str = config.DATASET_NAME, 
                           use_subgraph: bool = False, 
                           subgraph_size: int = 1000) -> Dict[str, Any]:
    if name == 'ogbn-arxiv':
        dataset = load_ogb_arxiv()
        dataset['node_texts'] = {i: f"Paper {i}" for i in range(dataset['num_nodes'])}
        dataset['edge_index'] = dataset['edge_index']
        dataset['adj_matrix'] = dataset['adj_matrix']
    elif name == 'cora':
        dataset = load_cora()
        edge_index = dataset['adj_matrix'].nonzero().t().contiguous()
        dataset['edge_index'] = edge_index
    elif name == 'chameleon':
        base_path = os.path.join(config.DATA_DIR, "chameleon")
        
        # 检查特征向量文件是否存在
        features_path = os.path.join(base_path, "features.pt")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"❌ Features file not found: {features_path}")
            
        # 加载特征向量
        features = torch.load(features_path)
        print(f"📊 Debug: Loaded features from {features_path}")
        print(f"📊 Debug: Features shape: {features.shape}")
        print(f"📊 Debug: Features device: {features.device}")
        print(f"📊 Debug: Features dtype: {features.dtype}")
        print(f"📊 Debug: Features sample: {features[0][:5]}...")
        
        # 确保特征向量在CPU上并且是float类型
        features = features.cpu().float()
        print("🔄 Debug: Converted features to CPU and float type")
        
        # 加载边索引
        edge_index = torch.load(os.path.join(base_path, "edge_index.pt"))
        
        # 加载标签
        labels = torch.load(os.path.join(base_path, "labels.pt"))
        
        # 加载训练/验证/测试掩码
        train_mask = torch.load(os.path.join(base_path, "train_mask.pt"))
        val_mask = torch.load(os.path.join(base_path, "val_mask.pt"))
        test_mask = torch.load(os.path.join(base_path, "test_mask.pt"))
        
        # 直接加载raw_texts.json
        with open(os.path.join(base_path, "raw_texts.json")) as f:
            text_list = json.load(f)
            node_texts = {i: text for i, text in enumerate(text_list)}
            
        adj_matrix = edge_index_to_adj(edge_index, len(features))
        train_idx = train_mask.nonzero(as_tuple=True)[0].tolist()
        val_idx = val_mask.nonzero(as_tuple=True)[0].tolist()
        test_idx = test_mask.nonzero(as_tuple=True)[0].tolist()
        
        print(f"📊 Debug: Dataset statistics:")
        print(f"  - Number of nodes: {len(features)}")
        print(f"  - Number of edges: {edge_index.size(1)}")
        print(f"  - Number of training nodes: {len(train_idx)}")
        print(f"  - Number of validation nodes: {len(val_idx)}")
        print(f"  - Number of test nodes: {len(test_idx)}")
        
        # 确保特征向量被正确传递
        if features is None:
            raise ValueError("❌ Features are None after loading")
            
        # 验证特征向量的形状
        if features.shape[0] != len(node_texts):
            raise ValueError(f"❌ Number of nodes in features ({features.shape[0]}) does not match number of texts ({len(node_texts)})")
            
        dataset = {
            "adj_matrix": adj_matrix,
            "edge_index": edge_index,
            "node_features": features,  # 确保特征向量被正确传递
            "node_texts": node_texts,
            "labels": labels,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "num_classes": len(set(labels.tolist())),
            "num_nodes": len(features)
        }
        
        # 验证特征向量是否被正确传递
        print(f"📊 Debug: Verifying features in dataset:")
        print(f"  - Features in dataset: {dataset['node_features'] is not None}")
        if dataset['node_features'] is not None:
            print(f"  - Features shape: {dataset['node_features'].shape}")
            print(f"  - Features device: {dataset['node_features'].device}")
            print(f"  - Features dtype: {dataset['node_features'].dtype}")
            print(f"  - Features sample: {dataset['node_features'][0][:5]}...")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset
