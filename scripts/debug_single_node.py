"""
Debug a single node's agent process inside Graph Agentic Network
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import os

from gan.graph import GraphAgenticNetwork
from gan.llm import MockLLMInterface
from data.dataset import load_or_create_dataset
import config
from gan.llm import MockLLMInterface, RemoteLLMInterface


def debug_single_node(node_id: int, layer: int, subgraph_size: int = 100):
    print(f"Loading dataset '{config.DATASET_NAME}' (subgraph_size={subgraph_size})...")
    dataset = load_or_create_dataset(config.DATASET_NAME, use_subgraph=True, subgraph_size=subgraph_size)

    adj_matrix = dataset['adj_matrix']
    node_features = dataset['node_features']
    labels = dataset['labels']
    node_texts = dataset['node_texts']
    train_idx = dataset['train_idx']

    # Choose LLM backend
    if config.LLM_BACKEND == "mock":
        llm_interface = MockLLMInterface()
    elif config.LLM_BACKEND == "remote":
        llm_interface = RemoteLLMInterface(
            endpoint=config.REMOTE_LLM_ENDPOINT,
            model_name=config.LLM_MODEL
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND}")

    # Init GAN
    gan = GraphAgenticNetwork(
        adj_matrix=adj_matrix,
        node_texts=node_texts,
        llm_interface=llm_interface,
        labels=labels,
        train_idx=train_idx,
        node_features=node_features,
        num_layers=config.NUM_LAYERS
    )

    # Fetch node
    node = gan.graph.get_node(node_id)
    print(f"\nDebugging node {node_id}")
    print("-" * 60)
    print(f"Initial features: {node.state.feature_vector[:10].tolist() if hasattr(node.state, 'feature_vector') and node.state.feature_vector is not None else 'None'} ...")
    print(f"Initial label: {node.state.label}")
    print(f"Initial predicted_label: {node.state.predicted_label}")
    print(f"Initial memory: {node.state.memory}")
    print(f"Initial message queue: {node.state.message_queue}")
    print(f"Total neighbors: {len(gan.graph.get_neighbors(node_id))}")

    # Test RAG query
    print("\nTesting RAG query:")
    print("-" * 60)
    results = gan.graph.rag_query(node_id, top_k=5)
    if results:
        print(f"Found {len(results)} similar nodes:")
        for nid, info in results.items():
            print(f"Node {nid}:")
            print(f"  - Text: {info['text'][:100]}...")
            print(f"  - Label: {info['label']}")
            print(f"  - Similarity: {info['similarity_score']:.4f}")
    else:
        print("No similar nodes found")

    # Run step
    print(f"\n==> Running layer {layer} for node {node_id}...")
    gan.graph.run_layer(layer=layer, node_indices=[node_id])

    # After step
    print("\nAfter step:")
    print("-" * 60)
    print(f"Predicted label: {node.state.predicted_label}")
    print(f"Memory size: {len(node.state.memory)}")
    if node.state.memory:
        print(f"Last memory item: {node.state.memory[-1]}")
    print(f"Message queue (after clear): {node.state.message_queue}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", type=int, default=0, help="Node ID to debug")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to run")
    parser.add_argument("--subgraph_size", type=int, default=100, help="Subgraph size for fast debugging")

    args = parser.parse_args()

    debug_single_node(
        node_id=args.node_id,
        layer=args.layer,
        subgraph_size=args.subgraph_size
    )
