"""
Node agent implementation for the Graph Agentic Network
"""

import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from gan.actions import Action, RetrieveAction, BroadcastAction, UpdateAction, NoOpAction


@dataclass
class NodeState:
    """Represents the internal state of a node agent."""
    
    node_id: int
    features: torch.Tensor
    hidden_state: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None
    predicted_label: Optional[torch.Tensor] = None
    message_queue: List[Dict[str, Any]] = field(default_factory=list)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    layer_count: int = 0
    
    def __post_init__(self):
        """Initialize hidden state if not provided."""
        if self.hidden_state is None:
            self.hidden_state = torch.zeros_like(self.features)
    
    def add_message(self, from_node: int, message: torch.Tensor, layer: int):
        """
        Add a message to the queue.
        
        Args:
            from_node: ID of the sender node
            message: Message tensor
            layer: Current layer when the message was sent
        """
        self.message_queue.append({
            "from": from_node,
            "content": message,
            "layer": layer
        })
    
    def clear_messages(self):
        """Clear the message queue."""
        self.message_queue = []
    
    def increment_layer(self):
        """Increment the layer counter."""
        self.layer_count += 1


class NodeAgent:
    """Represents an agent in the graph, corresponding to a node."""
    
    def __init__(self, state: NodeState, llm_interface: 'LLMInterface'):
        """
        Initialize a node agent.
        
        Args:
            state: Initial state of the node
            llm_interface: Interface to the large language model
        """
        self.state = state
        self.llm = llm_interface
        self.retrieved_data = {}
    
    def step(self, graph: 'AgenticGraph', layer: int) -> None:
        """
        Perform one step of decision-making and action execution.
        
        Args:
            graph: The graph environment
            layer: Current layer number
        """
        # Prepare context
        context = self._prepare_context(graph)
        
        # Get decision from LLM
        decision = self.llm.decide_action(context)
        
        # Convert decision to action
        action = self._create_action(decision)
        
        # Execute action and store result
        if action:
            result = action.execute(self, graph)
            
            # Store action in memory
            self.state.memory.append({
                "layer": layer,
                "action": result.get("action", "unknown"),
                "result": result
            })
    
    def receive_message(self, from_node: int, message: torch.Tensor) -> None:
        """
        Receive a message from another node.
        
        Args:
            from_node: ID of the sender node
            message: Message tensor
        """
        self.state.add_message(from_node, message, self.state.layer_count)
    
    def _prepare_context(self, graph: 'AgenticGraph') -> Dict[str, Any]:
        """
        Prepare the context for LLM decision-making.
        
        Args:
            graph: The graph environment
            
        Returns:
            Dictionary containing context information
        """
        # Get neighbors
        neighbors = graph.get_neighbors(self.state.node_id)
        
        # Format feature tensor for LLM consumption
        features_list = self.state.features.tolist() 
        if len(features_list) > 10:  # Truncate if too large
            features_list = features_list[:10] + ["..."]
            
        # Prepare messages
        messages = []
        for msg in self.state.message_queue[-5:]:  # Last 5 messages only
            messages.append({
                "from": msg["from"],
                "content_preview": msg["content"].mean().item(),  # Just a preview
                "layer": msg["layer"]
            })
            
        # Prepare memory
        recent_memory = self.state.memory[-3:] if self.state.memory else []
            
        return {
            "node_id": self.state.node_id,
            "layer": self.state.layer_count,
            "features": features_list,
            "hidden_state": self.state.hidden_state.tolist() if len(self.state.hidden_state) < 10 else self.state.hidden_state[:10].tolist() + ["..."],
            "label": self.state.label.item() if self.state.label is not None else None,
            "predicted_label": self.state.predicted_label.item() if self.state.predicted_label is not None else None,
            "neighbors": neighbors,
            "total_neighbors": len(neighbors),
            "messages": messages,
            "total_messages": len(self.state.message_queue),
            "memory": recent_memory,
            "total_memory": len(self.state.memory),
            "retrieved_data": self.retrieved_data
        }
    
    def _create_action(self, decision: Dict[str, Any]) -> Optional[Action]:
        """
        Create an action based on the LLM decision.
        
        Args:
            decision: Decision dictionary from the LLM
            
        Returns:
            An action object or None for no-op
        """
        action_type = decision.get("action_type", "no_op")
        
        if action_type == "retrieve":
            target_nodes = decision.get("target_nodes", [])
            info_type = decision.get("info_type", "features")
            return RetrieveAction(target_nodes, info_type)
            
        elif action_type == "broadcast":
            target_nodes = decision.get("target_nodes", [])
            message_data = decision.get("message", [0.0])
            
            # Convert message to tensor
            if isinstance(message_data, list):
                message = torch.tensor(message_data, dtype=torch.float)
            else:
                # Default to a simple scalar message
                message = torch.tensor([float(message_data)], dtype=torch.float)
                
            return BroadcastAction(target_nodes, message)
            
        elif action_type == "update":
            updates = {}
            
            if "hidden_state" in decision:
                if isinstance(decision["hidden_state"], list):
                    updates["hidden_state"] = torch.tensor(decision["hidden_state"], dtype=torch.float)
            
            if "predicted_label" in decision:
                updates["predicted_label"] = decision["predicted_label"]
                
            if updates:
                return UpdateAction(updates)
        
        # Default to no-op
        return NoOpAction()
