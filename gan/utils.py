"""
Utility functions for the Graph Agentic Network
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc

import config


def seed_everything(seed: int = config.RANDOM_SEED):
    """
    Set seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_node_classification(predictions: torch.Tensor, labels: torch.Tensor, 
                                mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Evaluate node classification performance.
    
    Args:
        predictions: Tensor of predicted labels
        labels: Tensor of true labels
        mask: Optional mask to evaluate only specific nodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    
    # Handle potential multi-class case
    try:
        f1_micro = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='micro')
        f1_macro = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
    except:
        f1_micro = 0.0
        f1_macro = 0.0
    
    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }


def visualize_node_embeddings(embeddings: Dict[int, torch.Tensor], 
                             labels: Optional[torch.Tensor] = None,
                             title: str = "Node Embeddings Visualization",
                             save_path: Optional[str] = None):
    """
    Visualize node embeddings using t-SNE.
    
    Args:
        embeddings: Dictionary of node embeddings
        labels: Optional tensor of node labels for coloring
        title: Title for the plot
        save_path: Optional path to save the visualization
    """
    # Extract node IDs and embeddings
    node_ids = list(embeddings.keys())
    embedding_matrix = torch.stack([embeddings[nid] for nid in node_ids]).cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED)
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Use labels for coloring
        node_labels = [labels[nid].item() if nid < len(labels) else -1 for nid in node_ids]
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=node_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Class')
    else:
        # No labels available
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_graph(adj_matrix: torch.Tensor, 
                   node_colors: Optional[List] = None,
                   node_size: int = 50,
                   title: str = "Graph Visualization",
                   save_path: Optional[str] = None):
    """
    Visualize the graph.
    
    Args:
        adj_matrix: Adjacency matrix
        node_colors: Optional list of node colors
        node_size: Size of nodes in the visualization
        title: Title for the plot
        save_path: Optional path to save the visualization
    """
    # Convert adjacency matrix to NetworkX graph
    G = nx.from_numpy_array(adj_matrix.cpu().numpy())
    
    # Plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=config.RANDOM_SEED)
    
    if node_colors is not None:
        nx.draw(G, pos, node_color=node_colors, with_labels=False, 
               node_size=node_size, cmap=plt.cm.tab10)
    else:
        nx.draw(G, pos, with_labels=False, node_size=node_size)
        
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def compare_results(gan_metrics: Dict[str, float], 
                   baseline_metrics: Dict[str, float],
                   title: str = "Model Comparison",
                   save_path: Optional[str] = None):
    """
    Compare results between GAN and baseline models.
    
    Args:
        gan_metrics: Dictionary of GAN metrics
        baseline_metrics: Dictionary of baseline metrics
        title: Title for the plot
        save_path: Optional path to save the visualization
    """
    # Collect metrics
    metrics = sorted(set(gan_metrics.keys()) & set(baseline_metrics.keys()))
    gan_values = [gan_metrics[m] for m in metrics]
    baseline_values = [baseline_metrics[m] for m in metrics]
    
    # Bar width
    width = 0.35
    
    # Create plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    
    plt.bar(x - width/2, gan_values, width, label='GAN')
    plt.bar(x + width/2, baseline_values, width, label='Baseline')
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(x, metrics)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_labeled_examples(
    memory: List[Dict[str, Any]],
    top_k: int = 5,
    dedup: bool = True
) -> List[Dict[str, Any]]:
    seen_sources = set()
    examples = []

    for m in memory:
        if not m.get("label") or not m.get("text"):
            continue
        if dedup and m.get("source") in seen_sources:
            continue
        seen_sources.add(m.get("source"))
        examples.append({"label": m["label"], "text": m["text"]})
        if len(examples) >= top_k:
            break
    return examples


# def truncate_text(text: str, max_words: int = 20) -> str:
#     words = text.strip().split()
#     return ' '.join(words[:max_words]) + ('...' if len(words) > max_words else '')

from config import USE_TOKEN_TRUNCATE, MEMORY_MAX_WORDS
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")  # default fallback
except:
    tokenizer = None  # 在 DeepSeek 场景中可为空

def truncate_text(text: str, max_words: int = MEMORY_MAX_WORDS) -> str:
    if USE_TOKEN_TRUNCATE and tokenizer:
        tokens = tokenizer.tokenize(text)
        truncated = tokenizer.convert_tokens_to_string(tokens[:max_words])
        return truncated + ("..." if len(tokens) > max_words else "")
    else:
        words = text.strip().split()
        return ' '.join(words[:max_words]) + ('...' if len(words) > max_words else '')




def has_memory_entry(agent_or_state, result: Dict[str, Any]) -> bool:
    """
    Check if a memory entry with same source + label + text already exists.
    Supports more robust deduplication.
    """
    memory = agent_or_state.state.memory if hasattr(agent_or_state, "state") else agent_or_state.memory

    # 提取核心信息用于比对
    new_text = result.get("text", "")
    new_label = result.get("label", None)
    new_source = result.get("source", None)
    new_action = result.get("action", result.get("action_type", ""))

    for m in memory:
        # 基于 Broadcast、Retrieve、RAG 写入
        if m.get("action") in {"RetrieveExample", "BroadcastLabel", "RAGResult"}:
            if (
                m.get("text") == new_text and
                m.get("label") == new_label and
                m.get("source") == new_source
            ):
                return True

        # 对于 update/fallback update 动作的结果也可以排重（可选）
        if m.get("action") == new_action and m.get("result") == result:
            return True

    return False

