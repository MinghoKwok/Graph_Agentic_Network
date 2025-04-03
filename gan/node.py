"""
Node agent implementation for the Graph Agentic Network
"""

import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from gan.actions import Action, RetrieveAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE  # 从 config 模块导入 DEBUG_STEP_SUMMARY 和 DEBUG_MESSAGE_TRACE
from data.cora.label_vocab import label_vocab  # 自定义标签映射


@dataclass
class NodeState:
    """Represents the internal state of a node agent."""
    
    node_id: int
    text: str  # 取代 features
    label: Optional[torch.Tensor] = None
    predicted_label: Optional[torch.Tensor] = None
    message_queue: List[Dict[str, Any]] = field(default_factory=list)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    layer_count: int = 0
    
    def __post_init__(self):
        """Initialize hidden state if not provided."""
        pass  # 移除 hidden_state 初始化逻辑
    
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
        self.memory = {}  # {node_id: {"features": ..., "label": ..., "messages": [...], "source_layer": int}}
    


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
            self.state.memory.append({
                "layer": layer,
                "action": result.get("action", "unknown"),
                "result": result
            })

        # Print debug summaries
        if (DEBUG_STEP_SUMMARY or DEBUG_MESSAGE_TRACE) and self.state.memory:
            last = self.state.memory[-1]
            action_type = last.get("action", "unknown")
            result = last.get("result", {})
            pred_label = (
                self.state.predicted_label.item()
                if self.state.predicted_label is not None else None
            )

            print(f"\n🧠 Agent Step | Node {self.state.node_id} | Layer {layer}")
            print(f"  ├─ 🏷️  Action: {action_type}")
            print(f"  ├─ 🎯 Predicted Label: {pred_label}")
            print(f"  ├─ 🧠 Memory size: {len(self.state.memory)}")
            print(f"  └─ 👥 Total neighbors: {len(context.get('neighbors', []))}")

        # Print detailed message trace
        if DEBUG_MESSAGE_TRACE and self.state.memory:
            print(f"\n🔍 Message Trace | Node {self.state.node_id} | Layer {layer}")
            
            if action_type == "retrieve":
                targets = result.get("target_nodes", [])
                results = result.get("results", {})
                print(f"  📥 Retrieved from {len(targets)} neighbor(s):")
                for tid in targets:
                    if tid in results:
                        preview = results[tid]
                        preview_str = self._format_preview(preview)
                        print(f"    ↳ Node {tid} ✅ {preview_str}")
                    else:
                        print(f"    ↳ Node {tid} ⛔ not found")

            elif action_type == "broadcast":
                targets = result.get("target_nodes", [])
                message = result.get("message", None)
                print(f"  📤 Broadcasted to {len(targets)} node(s): {targets}")
                if message is not None:
                    preview = str(message[:5].tolist()) + ("..." if len(message) > 5 else "")
                    print(f"    ↳ Message: {preview} (dim={len(message)})")

            elif action_type == "update":
                updated = result.get("updated_fields", [])
                print(f"  🛠️  Updated fields: {updated}")

            else:
                print("  ⚠️  No message or state updates in this step.")
        

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
            "text": self.state.text,  # 添加这一行
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
            info_type = decision.get("info_type", "text")
            return RetrieveAction(target_nodes, info_type)

        elif action_type == "broadcast":
            target_nodes = decision.get("target_nodes", [])
            message_data = decision.get("message", [0.0])

            # Convert message to tensor
            if isinstance(message_data, list) and all(isinstance(x, (int, float)) for x in message_data):
                message = torch.tensor(message_data, dtype=torch.float)
            elif isinstance(message_data, (int, float)):
                message = torch.tensor([message_data], dtype=torch.float)
            else:
                # Fallback: encode string (or dict etc.) as its length
                message = torch.tensor([len(str(message_data))], dtype=torch.float)

            return BroadcastAction(target_nodes, message)

        elif action_type == "update":
            updates = {}

            if "predicted_label" in decision:
                label_str = decision.get("predicted_label")
                label_id = label_vocab.get(label_str, -1)
                if label_id != -1:
                    updates["predicted_label"] = torch.tensor(label_id)

            if updates:
                return UpdateAction(updates)

        # Default to no-op
        return NoOpAction()

    def _format_preview(self, obj: Any, max_len: int = 60) -> str:
        """
        Format a preview string from any object for display.
        
        Args:
            obj: The object to preview
            max_len: Max number of characters
        
        Returns:
            Truncated string representation
        """
        try:
            if isinstance(obj, torch.Tensor):
                return str(obj.tolist()[:5]) + ("..." if obj.numel() > 5 else "")
            elif isinstance(obj, (list, tuple)):
                return str(obj[:5]) + ("..." if len(obj) > 5 else "")
            elif isinstance(obj, dict):
                keys = list(obj.keys())[:3]
                preview = {k: obj[k] for k in keys}
                return str(preview) + ("..." if len(obj) > 3 else "")
            elif isinstance(obj, (str, int, float, bool)):
                return str(obj)[:max_len] + ("..." if len(str(obj)) > max_len else "")
            elif obj is None:
                return "None"
            else:
                return str(type(obj))  # fallback: just print type name
        except Exception as e:
            return f"[Preview error: {e}]"

