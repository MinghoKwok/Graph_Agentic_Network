"""
Graph structure and network implementation for Graph Agentic Network
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
import json

import config
from gan.node import NodeState, NodeAgent


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AgenticGraph:
    """Graph structure for agentic network."""

    def __init__(self, adj_matrix, llm_interface, labels=None, train_idx=None):
        """
        Initialize the graph with adjacency matrix and node agents.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            llm_interface: LLM interface for node agents
            labels: Optional tensor of node labels
            train_idx: Optional tensor of training node indices
        """
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        
        # Load node texts from JSONL file
        with open(f"../data/{config.DATASET_NAME}/cora_text_graph_simplified.jsonl") as f:
            records = [json.loads(l) for l in f]
        node_texts = {r["node_id"]: r["text"] for r in records}
        self._node_texts = node_texts
        
        # Create train mask if train_idx is provided
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        if train_idx is not None:
            train_mask[train_idx] = True
            
        # Initialize node agents
        self.nodes = {}
        for i in range(self.num_nodes):
            # Only keep labels for training nodes
            if labels is not None and train_mask[i]:
                node_label = torch.tensor(int(labels[i]))
            else:
                node_label = None
            text = node_texts.get(i, "")
            state = NodeState(
                node_id=i,
                text=text,
                label=node_label
            )
            self.nodes[i] = NodeAgent(state, llm_interface)

        # 初始化向量索引
        self._initialize_vector_index()

    def _initialize_vector_index(self):
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        import os

        emb_path = f"../data/{config.DATASET_NAME}/cora_text_embeddings.npy"
        if not os.path.exists(emb_path):
            print(f"❌ Missing embedding file: {emb_path}")
            self._node_features_index = None
            return

        # ✅ 加载 embedding
        node_embeddings = np.load(emb_path, allow_pickle=True).item()
        self._node_id_to_query_vector = node_embeddings
        print(f"✅ Loaded sentence-transformer embeddings for {len(node_embeddings)} nodes")

        # ✅ 构建索引（仅训练集节点）
        node_features = []
        node_indices = []
        for node_id, agent in self.nodes.items():
            if agent.state.label is not None:
                vec = node_embeddings.get(node_id)
                if vec is not None:
                    node_features.append(vec)
                    node_indices.append(node_id)

        self._feature_idx_to_node_id = {i: nid for i, nid in enumerate(node_indices)}

        if len(node_features) > 0:
            self._node_features_index = NearestNeighbors(
                n_neighbors=min(10, len(node_features)),
                metric='cosine'
            )
            self._node_features_index.fit(np.stack(node_features))
            print(f"✅ RAG index built with {len(node_features)} labeled candidates")
        else:
            print("⚠️ No valid labeled nodes for RAG index")
            self._node_features_index = None


    def rag_query(self, query_id: int, top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        try:
            node_id = query_id
            query_vector = self._node_id_to_query_vector.get(node_id)
            if query_vector is None:
                print(f"⚠️ Node {node_id} has no query vector.")
                return {}
            query_vector = query_vector.reshape(1, -1)
        except (ValueError, TypeError):
            print(f"⚠️ RAG query must be a node ID. Got: {query}")
            return {}

        if self._node_features_index is None:
            print("⚠️ Vector index not available for RAG query")
            return {}

        distances, indices = self._node_features_index.kneighbors(
            query_vector, n_neighbors=min(top_k, self._node_features_index.n_samples_fit_)
        )

        results = {}
        for i, idx in enumerate(indices[0]):
            result_node_id = self._feature_idx_to_node_id.get(idx)
            if result_node_id is not None:
                node = self.get_node(result_node_id)
                if node and node.state.label is not None:
                    results[result_node_id] = {
                        'text': self._node_texts.get(result_node_id, ""),
                        'label': node.state.label.item(),
                        'similarity_score': -float(distances[0][i])
                    }
        
        print(f"🔍 RAG Query: Node {node_id}")
        print(f"📊 Candidate pool size: {len(self._feature_idx_to_node_id)}")
        print(f"📎 Top-{top_k} results:")
        for result_id, result in results.items():
            print(f"  ↳ Node {result_id} | Label: {result['label']} | Score: {result['similarity_score']:.3f}")

        return results

    def has_node(self, node_id: int) -> bool:
        """
        Check whether the node exists in the graph.
        
        Args:
            node_id: Node ID to check
        
        Returns:
            True if node exists, False otherwise
        """
        return node_id in self.nodes

    def get_node(self, node_id: int) -> NodeAgent:
        """
        Get a node agent by ID.
        
        Args:
            node_id: ID of the node
            
        Returns:
            The node agent object
        """
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of neighbor node IDs
        """
        neighbors = torch.where(self.adj_matrix[node_id] > 0)[0].tolist()
        
        # Limit number of neighbors if too many
        if len(neighbors) > config.MAX_NEIGHBORS:
            neighbors = neighbors[:config.MAX_NEIGHBORS]
            
        return neighbors
    
    def run_layer(self, layer: int, node_indices: Optional[List[int]] = None) -> None:
        """
        Run one layer of agent processing on the graph.
        
        Args:
            layer: Layer number
            node_indices: Optional subset of nodes to process (for batching)
        """
        # Process only specified nodes or all nodes
        if node_indices is None:
            node_indices = list(self.nodes.keys())
        
        for node_id in tqdm(node_indices, desc=f"Processing layer {layer}"):
            agent = self.nodes[node_id]
            agent.step(self, layer)
        
        # After all nodes have executed actions, prepare for next layer
        for node_id in node_indices:
            agent = self.nodes[node_id]
            agent.state.increment_layer()
            agent.state.clear_messages()  # Clear messages to avoid memory buildup

    def initialize_labels_and_index(self, train_idx: List[int], labels: torch.Tensor):
        """
        初始化训练集标签并构建向量索引。
        
        Args:
            train_idx: 训练集节点ID列表
            labels: 节点标签张量
        """
        # 写入训练集标签
        for nid in train_idx:
            if nid in self.nodes:
                self.nodes[nid].state.label = int(labels[nid])
        
        # 初始化向量索引
        self._initialize_vector_index()


class GraphAgenticNetwork:
    """Main class for the Graph Agentic Network framework."""
    
    def __init__(self, 
                adj_matrix: torch.Tensor, 
                node_texts: List[str],
                llm_interface: 'LLMInterface',
                labels: Optional[torch.Tensor] = None,
                num_layers: int = config.NUM_LAYERS,
                train_idx: Optional[List[int]] = None):  # ✅ 添加 train_idx 参数
        """
        Initialize the Graph Agentic Network.

        Args:
            adj_matrix: Adjacency matrix of shape (num_nodes, num_nodes)
            node_texts: List of node texts
            llm_interface: Interface to the large language model
            labels: Optional tensor of node labels
            num_layers: Number of layers to process
            train_idx: Optional list of training node indices
        """
        self.graph = AgenticGraph(
            adj_matrix, llm_interface, labels=labels, train_idx=train_idx  # ✅ 传入 AgenticGraph
        )
        self.num_layers = num_layers
        self.llm_interface = llm_interface
        self.current_layer = 0
    
    def forward(self, batch_size: Optional[int] = None) -> None:
        """
        Run the Graph Agentic Network for the specified number of layers.
        
        Args:
            batch_size: Optional batch size for processing nodes
        """
        all_nodes = list(range(self.graph.num_nodes))
        
        for layer in range(self.num_layers):
            self.current_layer = layer
            print(f"Processing layer {layer+1}/{self.num_layers}")
            
            # Process in batches if specified
            if batch_size is not None and batch_size < self.graph.num_nodes:
                # Divide nodes into batches
                num_batches = (self.graph.num_nodes + batch_size - 1) // batch_size
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, self.graph.num_nodes)
                    batch_nodes = all_nodes[start_idx:end_idx]
                    
                    self.graph.run_layer(layer, batch_nodes)
            else:
                # Process all nodes at once
                self.graph.run_layer(layer)
    
    def get_node_predictions(self) -> torch.Tensor:
        """
        Get the node predictions (e.g., for node classification).
        
        Returns:
            Tensor of predicted labels
        """
        predictions = torch.zeros(self.graph.num_nodes, dtype=torch.long)
        
        for node_id, agent in self.graph.nodes.items():
            if agent.state.predicted_label is not None:
                predictions[node_id] = agent.state.predicted_label
        
        return predictions
    
    def get_node_representations(self) -> Dict[int, torch.Tensor]:
        """
        Get the final node representations.
        
        Returns:
            Dictionary mapping node IDs to their hidden state
        """
        return {node_id: agent.state.hidden_state.clone() 
                for node_id, agent in self.graph.nodes.items()}
    
    def get_node_memory(self) -> Dict[int, List[Dict]]:
        """
        Get the memory of each node.
        
        Returns:
            Dictionary mapping node IDs to their memory list
        """
        return {node_id: agent.state.memory.copy() 
                for node_id, agent in self.graph.nodes.items()}
