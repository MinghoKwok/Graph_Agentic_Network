"""
Graph structure and network implementation for Graph Agentic Network
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm

import config
from gan.node import NodeState, NodeAgent


class AgenticGraph:
    """Represents the graph with node agents."""
    
    def __init__(self, adj_matrix: torch.Tensor, node_features: torch.Tensor, 
                 llm_interface: 'LLMInterface', labels: Optional[torch.Tensor] = None):
        """
        Initialize the agentic graph.
        
        Args:
            adj_matrix: Adjacency matrix of shape (num_nodes, num_nodes)
            node_features: Feature matrix of shape (num_nodes, feature_dim)
            llm_interface: Interface to the large language model
            labels: Optional tensor of node labels
        """
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        
        # Initialize node agents
        self.nodes = {}
        for i in range(self.num_nodes):
            node_label = labels[i] if labels is not None else None
            state = NodeState(
                node_id=i,
                features=node_features[i],
                label=node_label
            )
            self.nodes[i] = NodeAgent(state, llm_interface)
    
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


class GraphAgenticNetwork:
    """Main class for the Graph Agentic Network framework."""
    
    def __init__(self, adj_matrix: torch.Tensor, node_features: torch.Tensor,
                 llm_interface: 'LLMInterface', labels: Optional[torch.Tensor] = None,
                 num_layers: int = config.NUM_LAYERS):
        """
        Initialize the Graph Agentic Network.
        
        Args:
            adj_matrix: Adjacency matrix of shape (num_nodes, num_nodes)
            node_features: Feature matrix of shape (num_nodes, feature_dim)
            llm_interface: Interface to the large language model
            labels: Optional tensor of node labels
            num_layers: Number of layers to process
        """
        self.graph = AgenticGraph(adj_matrix, node_features, llm_interface, labels)
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
