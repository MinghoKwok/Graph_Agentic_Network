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

    def __init__(self, adj_matrix, llm_interface, node_texts: Dict[int, str], labels=None, train_idx=None, node_features: Optional[torch.Tensor] = None):
        print("\n🔍 Debug: AgenticGraph initialization")
        
        # 1. 基本属性初始化
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        self._node_texts = node_texts
        self.train_idx = train_idx if train_idx is not None else []
        self._node_id_to_query_vector = {}
        self._node_features_index = None
        self._feature_idx_to_node_id = {}
        
        # 2. 特征向量预处理
        if node_features is not None:
            print(f"📊 Debug: Processing node features:")
            print(f"  - Original shape: {node_features.shape}")
            print(f"  - Original device: {node_features.device}")
            print(f"  - Original dtype: {node_features.dtype}")
            
            # 确保特征向量在CPU上并且是float类型
            node_features = node_features.cpu().float()
            print(f"  - Converted to CPU and float type")
            print(f"  - Final shape: {node_features.shape}")
            print(f"  - Final dtype: {node_features.dtype}")
            print(f"  - Sample: {node_features[0][:5]}...")
        else:
            print("⚠️ Warning: No node features provided")
            
        # 3. 节点初始化
        self.nodes = {}
        for i in range(self.num_nodes):
            # 创建节点状态
            state = NodeState(
                node_id=i,
                text=node_texts.get(i, ""),
                label=None
            )
            
            # 分配特征向量
            if node_features is not None:
                state.feature_vector = node_features[i].clone()
                if i < 5:  # 只打印前5个节点的信息
                    print(f"✅ Node {i}: Feature vector assigned")
                    print(f"  - Shape: {state.feature_vector.shape}")
                    print(f"  - Dtype: {state.feature_vector.dtype}")
                    print(f"  - Sample: {state.feature_vector[:5]}...")
            
            # 创建节点代理
            self.nodes[i] = NodeAgent(state, llm_interface)
            
        # 4. 初始化RAG索引
        self._initialize_vector_index()
        
    def _initialize_vector_index(self):
        """初始化向量索引"""
        if config.DATASET_NAME == "cora":
            self._initialize_vector_index_from_file()
        else:
            self._initialize_vector_index_from_tensor()
            
    def _initialize_vector_index_from_file(self):
        from sklearn.neighbors import NearestNeighbors
        emb_path = f"../data/{config.DATASET_NAME}/cora_text_embeddings.npy"
        if not os.path.exists(emb_path):
            print(f"❌ Missing embedding file: {emb_path}")
            self._node_features_index = None
            return

        node_embeddings = np.load(emb_path, allow_pickle=True).item()
        self._node_id_to_query_vector = node_embeddings
        print(f"✅ Loaded sentence-transformer embeddings for {len(node_embeddings)} nodes")

        node_features, node_indices = [], []
        for node_id, agent in self.nodes.items():
            if agent.state.label is not None:
                vec = node_embeddings.get(node_id)
                if vec is not None:
                    node_features.append(vec)
                    node_indices.append(node_id)

        self._feature_idx_to_node_id = {i: nid for i, nid in enumerate(node_indices)}
        if node_features:
            self._node_features_index = NearestNeighbors(n_neighbors=min(10, len(node_features)), metric='cosine')
            self._node_features_index.fit(np.stack(node_features))
            print(f"✅ RAG index built with {len(node_features)} labeled candidates")
        else:
            print("⚠️ No valid labeled nodes for RAG index")
            self._node_features_index = None

    def _initialize_vector_index_from_tensor(self):
        """从张量初始化向量索引"""
        from sklearn.neighbors import NearestNeighbors
        print("\n🔍 Debug: Initializing vector index from tensor")
        
        # 1. 准备训练索引
        if isinstance(self.train_idx, torch.Tensor):
            self.train_idx = self.train_idx.tolist()
        self.train_idx = list(set(self.train_idx))
        print(f"📊 Debug: Processed train_idx: {len(self.train_idx)} unique indices")
        
        # 2. 收集有效的训练节点
        valid_nodes = []
        for node_id in self.train_idx:
            node = self.nodes.get(node_id)
            if node is None:
                continue
                
            # 检查节点是否满足所有条件
            has_label = node.state.label is not None
            has_feature = hasattr(node.state, 'feature_vector') and node.state.feature_vector is not None
            
            if has_label and has_feature:
                valid_nodes.append(node_id)
                
        print(f"📊 Debug: Found {len(valid_nodes)} valid training nodes")
        
        # 3. 构建查询向量映射
        for node_id, node in self.nodes.items():
            if hasattr(node.state, 'feature_vector') and node.state.feature_vector is not None:
                feature_vector = node.state.feature_vector.cpu().numpy()
                self._node_id_to_query_vector[node_id] = feature_vector
                
        print(f"📊 Debug: Created query vectors for {len(self._node_id_to_query_vector)} nodes")
        
        # 4. 构建RAG索引
        if valid_nodes:
            # 收集特征向量
            node_features = []
            node_indices = []
            
            for node_id in valid_nodes:
                node = self.nodes[node_id]
                feature_vector = node.state.feature_vector.cpu().numpy()
                node_features.append(feature_vector)
                node_indices.append(node_id)
                
            # 创建特征索引映射
            self._feature_idx_to_node_id = {i: nid for i, nid in enumerate(node_indices)}
            
            # 构建最近邻索引
            self._node_features_index = NearestNeighbors(
                n_neighbors=min(10, len(node_features)),
                metric='cosine'
            )
            self._node_features_index.fit(np.stack(node_features))
            
            print(f"✅ RAG index built with {len(node_features)} labeled nodes")
        else:
            print("⚠️ No valid nodes for RAG index")
            self._node_features_index = None
            
    def rag_query(self, query_id: int, top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        """执行RAG查询"""
        if query_id not in self._node_id_to_query_vector:
            print(f"⚠️ Node {query_id} has no query vector")
            return {}
            
        if self._node_features_index is None:
            print("⚠️ RAG index not available")
            return {}
            
        # 获取查询向量
        query_vector = self._node_id_to_query_vector[query_id]
        
        # 执行最近邻搜索
        distances, indices = self._node_features_index.kneighbors(
            query_vector.reshape(1, -1),
            n_neighbors=min(top_k, self._node_features_index.n_samples_fit_)
        )
        
        # 收集结果
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
                    
        return results
        
    def initialize_labels_and_index(self, train_idx: List[int], labels: torch.Tensor):
        """初始化标签并重建RAG索引"""
        print("\n🔍 Debug: Initializing labels")
        
        # 1. 确保输入类型正确
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if isinstance(train_idx, torch.Tensor):
            train_idx = train_idx.tolist()
            
        # 2. 设置标签
        for nid in train_idx:
            if nid in self.nodes:
                label_value = int(labels[nid])
                self.nodes[nid].state.label = torch.tensor(label_value)
                
        print(f"✅ Set labels for {len(train_idx)} nodes")
        
        # 3. 重建RAG索引
        self._initialize_vector_index()
        
    def get_node(self, node_id: int) -> Optional[NodeAgent]:
        """获取节点代理"""
        return self.nodes.get(node_id)
        
    def get_neighbors(self, node_id: int) -> List[int]:
        """获取节点的邻居"""
        neighbors = torch.where(self.adj_matrix[node_id] > 0)[0].tolist()
        if len(neighbors) > config.MAX_NEIGHBORS:
            neighbors = neighbors[:config.MAX_NEIGHBORS]
        return neighbors
        
    def run_layer(self, layer: int, node_indices: Optional[List[int]] = None) -> None:
        """运行一层消息传递"""
        if node_indices is None:
            node_indices = list(self.nodes.keys())
            
        # 处理每个节点
        for node_id in tqdm(node_indices, desc=f"Processing layer {layer}"):
            self.nodes[node_id].step(self, layer)
            
        # 清理消息队列
        for node_id in node_indices:
            self.nodes[node_id].state.increment_layer()
            self.nodes[node_id].state.clear_messages()


class GraphAgenticNetwork:
    def __init__(self, 
                adj_matrix: torch.Tensor, 
                node_texts: Dict[int, str],
                llm_interface: 'LLMInterface',
                labels: Optional[torch.Tensor] = None,
                num_layers: int = config.NUM_LAYERS,
                train_idx: Optional[List[int]] = None,
                node_features: Optional[torch.Tensor] = None):
        """
        Initialize the graph agentic network.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            node_texts: Dictionary mapping node IDs to their text descriptions
            llm_interface: Interface to the large language model
            labels: Optional tensor of node labels
            num_layers: Number of message passing layers
            train_idx: Optional list of training node indices
            node_features: Optional tensor of node features
        """
        print("\n🔍 Debug: GraphAgenticNetwork initialization")
        print(f"📊 Debug: train_idx type: {type(train_idx)}")
        if train_idx is not None:
            print(f"📊 Debug: train_idx content: {train_idx[:5]}... (showing first 5 elements)")
        print(f"📊 Debug: node_features type: {type(node_features)}")
        if node_features is not None:
            print(f"📊 Debug: node_features shape: {node_features.shape}")
            print(f"📊 Debug: node_features device: {node_features.device}")
            print(f"📊 Debug: node_features dtype: {node_features.dtype}")
            print(f"📊 Debug: node_features sample: {node_features[0][:5]}...")
            
        # 确保train_idx是列表类型
        if train_idx is not None and isinstance(train_idx, torch.Tensor):
            train_idx = train_idx.tolist()
            print("🔄 Debug: Converted train_idx from tensor to list")
            
        # 确保node_features在CPU上并且是float类型
        if node_features is not None:
            node_features = node_features.cpu().float()
            print("🔄 Debug: Converted node_features to CPU and float type")
        else:
            print("⚠️ Warning: node_features is None")
            
        # 验证特征向量的形状
        if node_features is not None:
            if node_features.shape[0] != adj_matrix.shape[0]:
                raise ValueError(f"❌ Number of nodes in features ({node_features.shape[0]}) does not match adjacency matrix ({adj_matrix.shape[0]})")
            print(f"✅ Verified feature dimensions match graph size")
            
        # 验证特征向量的形状与node_texts匹配
        if node_features is not None and len(node_texts) != node_features.shape[0]:
            raise ValueError(f"❌ Number of nodes in features ({node_features.shape[0]}) does not match number of texts ({len(node_texts)})")
            
        self.graph = AgenticGraph(
            adj_matrix=adj_matrix,
            llm_interface=llm_interface,
            node_texts=node_texts,
            labels=None,  # 初始化时不传递labels
            train_idx=train_idx,
            node_features=node_features  # 确保特征向量被传递
        )
        self.num_layers = num_layers
        self.llm_interface = llm_interface
        self.current_layer = 0
        
        # 在初始化完成后，统一设置标签并构建RAG索引
        if labels is not None and train_idx is not None:
            print("\n🔍 Debug: Initializing labels and building RAG index")
            self.graph.initialize_labels_and_index(train_idx, labels)

    def forward(self, batch_size: Optional[int] = None) -> None:
        all_nodes = list(range(self.graph.num_nodes))
        for layer in range(self.num_layers):
            self.current_layer = layer
            print(f"Processing layer {layer+1}/{self.num_layers}")
            if batch_size is not None and batch_size < self.graph.num_nodes:
                num_batches = (self.graph.num_nodes + batch_size - 1) // batch_size
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, self.graph.num_nodes)
                    batch_nodes = all_nodes[start_idx:end_idx]
                    self.graph.run_layer(layer, batch_nodes)
            else:
                self.graph.run_layer(layer)

    def get_node_predictions(self) -> torch.Tensor:
        predictions = torch.zeros(self.graph.num_nodes, dtype=torch.long)
        for node_id, agent in self.graph.nodes.items():
            if agent.state.predicted_label is not None:
                predictions[node_id] = agent.state.predicted_label
        return predictions

    def get_node_representations(self) -> Dict[int, torch.Tensor]:
        return {node_id: agent.state.hidden_state.clone() 
                for node_id, agent in self.graph.nodes.items()}

    def get_node_memory(self) -> Dict[int, List[Dict]]:
        return {node_id: agent.state.memory.copy() 
                for node_id, agent in self.graph.nodes.items()}
