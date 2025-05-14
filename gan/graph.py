"""
Graph structure and network implementation for Graph Agentic Network
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
import json
import os

import config
from gan.node import NodeState, NodeAgent


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AgenticGraph:
    """Graph structure for agentic network."""

    def __init__(self, adj_matrix, llm_interface, labels=None, train_idx=None, dataset_name=None):
        """
        Initialize the graph with adjacency matrix and node agents.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            llm_interface: LLM interface for node agents
            labels: Optional tensor of node labels
            train_idx: Optional tensor of training node indices
            dataset_name: Name of the dataset (defaults to config.DATASET_NAME)
        """
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        self.dataset_name = dataset_name or config.DATASET_NAME
        
        # Load node texts from JSONL file
        dataset_config = config.DATASET_CONFIGS[self.dataset_name]
        jsonl_path = os.path.join("..", "data", self.dataset_name, dataset_config["text_graph_file"])
        with open(jsonl_path) as f:
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

        # Initialize vector index
        self._initialize_vector_index()

    def _initialize_vector_index(self):
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        import os
        # import pdb; pdb.set_trace()

        dataset_config = config.DATASET_CONFIGS[self.dataset_name]
        emb_path = os.path.join("..", "data", self.dataset_name, dataset_config["embeddings_file"])
        if not os.path.exists(emb_path):
            print(f"❌ Missing embedding file: {emb_path}")
            self._node_features_index = None
            return

        # Load embeddings
        print(f"[DEBUG] 正在加载 embedding 文件: {emb_path}")
        node_embeddings = np.load(emb_path, allow_pickle=True).item()
        
        # [DEBUG] 检查 embedding 数据结构
        if not isinstance(node_embeddings, dict):
            print(f"[DEBUG] ⚠️ 警告: embeddings 不是 dict 类型，而是 {type(node_embeddings)}")
            # 尝试转换
            if hasattr(node_embeddings, 'item'):
                try:
                    node_embeddings = node_embeddings.item()
                    if not isinstance(node_embeddings, dict):
                        print(f"[DEBUG] ⚠️ 转换后仍不是 dict: {type(node_embeddings)}")
                except:
                    print(f"[DEBUG] ❌ 转换 embeddings 到 dict 失败")
        
        # [DEBUG] 检查 embedding 内容
        print(f"[DEBUG] Embedding 键类型: {type(list(node_embeddings.keys())[0]) if node_embeddings and len(node_embeddings) > 0 else 'N/A'}")
        print(f"[DEBUG] Embedding 值类型: {type(list(node_embeddings.values())[0]) if node_embeddings and len(node_embeddings) > 0 else 'N/A'}")
        print(f"[DEBUG] Embedding 值形状: {list(node_embeddings.values())[0].shape if node_embeddings and len(node_embeddings) > 0 and hasattr(list(node_embeddings.values())[0], 'shape') else 'N/A'}")
        
        # [DEBUG] 检查 embedding 键值的范围
        if node_embeddings and len(node_embeddings) > 0:
            min_key = min(node_embeddings.keys())
            max_key = max(node_embeddings.keys())
            print(f"[DEBUG] Embedding 键范围: [{min_key}, {max_key}]")
            
            # 检查数值范围是否合理
            sample_vec = list(node_embeddings.values())[0]
            if hasattr(sample_vec, 'min') and hasattr(sample_vec, 'max'):
                print(f"[DEBUG] Embedding 值范围: [{sample_vec.min():.4f}, {sample_vec.max():.4f}]")
                print(f"[DEBUG] Embedding 平均值: {sample_vec.mean():.4f}")
        
        self._node_id_to_query_vector = node_embeddings
        print(f"✅ Loaded sentence-transformer embeddings for {len(node_embeddings)} nodes")

        # Build index (only for training nodes)
        node_features = []
        node_indices = []
        
        # [DEBUG] 检查训练节点数量
        labeled_nodes_count = sum(1 for agent in self.nodes.values() if agent.state.label is not None)
        print(f"[DEBUG] 有标签的训练节点数量: {labeled_nodes_count}")
        
        missing_embedding_count = 0
        node_id_mismatch_count = 0
        for node_id, agent in self.nodes.items():
            if agent.state.label is not None:
                # 检查节点ID是否在embedding键中存在
                if node_id not in node_embeddings:
                    node_id_mismatch_count += 1
                    if node_id_mismatch_count <= 5:
                        print(f"[DEBUG] ⚠️ 节点ID不匹配: {node_id} 不在embedding键中")
                    
                vec = node_embeddings.get(node_id)
                if vec is not None:
                    node_features.append(vec)
                    node_indices.append(node_id)
                else:
                    missing_embedding_count += 1
                    if missing_embedding_count <= 5:
                        print(f"[DEBUG] ⚠️ 节点 {node_id} 缺少embedding向量")
        
        if missing_embedding_count > 0:
            print(f"[DEBUG] ⚠️ {missing_embedding_count} 个有标签节点缺少 embedding")
        if node_id_mismatch_count > 0:
            print(f"[DEBUG] ⚠️ {node_id_mismatch_count} 个有标签节点的ID不在embedding键中")
        
        self._feature_idx_to_node_id = {i: nid for i, nid in enumerate(node_indices)}

        if len(node_features) > 0:
            print(f"[DEBUG] 构建NearestNeighbors索引，使用 {len(node_features)} 个向量")
            
            # 检查特征向量是否有NaN或异常值
            stacked_features = np.stack(node_features)
            if np.isnan(stacked_features).any():
                print(f"[DEBUG] ⚠️ 特征向量中包含NaN值!")
            if np.isinf(stacked_features).any():
                print(f"[DEBUG] ⚠️ 特征向量中包含Inf值!")
            
            # 检查向量归一化
            norms = np.linalg.norm(stacked_features, axis=1)
            print(f"[DEBUG] 向量范数统计: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
            
            # 如果向量范数差异过大，可以考虑归一化
            if norms.max() / norms.min() > 10:
                print(f"[DEBUG] ⚠️ 向量范数差异较大，可能影响余弦相似度计算")
                print(f"[DEBUG] 考虑对向量进行归一化处理")
            
            self._node_features_index = NearestNeighbors(
                n_neighbors=min(10, len(node_features)),
                metric='cosine'
            )
            self._node_features_index.fit(stacked_features)
            print(f"✅ RAG index built with {len(node_features)} labeled candidates")
        else:
            print("⚠️ No valid labeled nodes for RAG index")
            self._node_features_index = None


    def rag_query(self, query_id: int, top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        # import pdb; pdb.set_trace()
        try:
            node_id = query_id
            query_vector = self._node_id_to_query_vector.get(node_id)
            if query_vector is None:
                print(f"⚠️ Node {node_id} has no query vector.")
                return {}
            
            # [DEBUG] 打印查询向量信息
            print(f"[DEBUG] Query vector for node {node_id}:")
            print(f"[DEBUG] Vector shape: {query_vector.shape}, Type: {type(query_vector)}")
            print(f"[DEBUG] Vector norm: {np.linalg.norm(query_vector):.4f}")
            
            query_vector = query_vector.reshape(1, -1)
        except (ValueError, TypeError) as e:
            print(f"⚠️ RAG query error: {e}")
            print(f"⚠️ RAG query must be a node ID. Got: {query_id}")
            return {}

        if self._node_features_index is None:
            print("⚠️ Vector index not available for RAG query")
            return {}

        distances, indices = self._node_features_index.kneighbors(
            query_vector, n_neighbors=min(top_k, self._node_features_index.n_samples_fit_)
        )
        
        # [DEBUG] 打印索引内容信息
        print(f"[DEBUG] Nearest neighbors index info:")
        print(f"[DEBUG] Index size: {self._node_features_index.n_samples_fit_}")
        print(f"[DEBUG] Feature map size: {len(self._feature_idx_to_node_id)}")

        results = {}
        print(f"[DEBUG] 详细相似度排名:")
        for i, idx in enumerate(indices[0]):
            result_node_id = self._feature_idx_to_node_id.get(idx)
            if result_node_id is not None:
                node = self.get_node(result_node_id)
                similarity_score = -float(distances[0][i])
                print(f"[DEBUG]   Rank {i+1}: Node {result_node_id} | Score: {similarity_score:.4f}")
                if node and node.state.label is not None:
                    result_vector = self._node_id_to_query_vector.get(result_node_id)
                    if result_vector is not None:
                        # 额外打印向量信息和相似度计算细节
                        cos_sim = np.dot(query_vector.flatten(), result_vector.flatten()) / (np.linalg.norm(query_vector) * np.linalg.norm(result_vector))
                        print(f"[DEBUG]   - Label: {node.state.label.item()} | Vector norm: {np.linalg.norm(result_vector):.4f} | Cosine sim: {cos_sim:.4f}")
                    
                    results[result_node_id] = {
                        'text': self._node_texts.get(result_node_id, ""),
                        'label': node.state.label.item(),
                        'similarity_score': similarity_score
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
                train_idx: Optional[List[int]] = None,
                dataset_name: str = config.DATASET_NAME):  # 添加 dataset_name 参数
        """
        Initialize the Graph Agentic Network.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            node_texts: List of node text descriptions
            llm_interface: LLM interface for node agents
            labels: Optional tensor of node labels
            num_layers: Number of GAN layers to run
            train_idx: Optional list of training node indices
            dataset_name: Name of the dataset (defaults to config.DATASET_NAME)
        """
        self.graph = AgenticGraph(
            adj_matrix=adj_matrix,
            llm_interface=llm_interface,
            labels=labels,
            train_idx=train_idx,
            dataset_name=dataset_name
        )
        self.num_layers = num_layers
        self.train_idx = train_idx
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
