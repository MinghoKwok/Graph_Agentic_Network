"""
Node agent implementation for the Graph Agentic Network
"""

import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from gan.actions import Action, RetrieveAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE, NUM_LAYERS  # 加入 NUM_LAYERS 以判断是否为最后一层
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
        pass

    def add_message(self, from_node: int, message: torch.Tensor, layer: int):
        self.message_queue.append({"from": from_node, "content": message, "layer": layer})

    def clear_messages(self):
        self.message_queue = []

    def increment_layer(self):
        self.layer_count += 1


class NodeAgent:
    """Represents an agent in the graph, corresponding to a node."""

    def __init__(self, state: NodeState, llm_interface: 'LLMInterface'):
        self.state = state
        self.llm = llm_interface
        self.retrieved_data = {}
        self.memory = {}

    def step(self, graph: 'AgenticGraph', layer: int) -> None:
        context = self._prepare_context(graph)
        decisions = self.llm.decide_action(context)

        # Ensure decisions is a list
        if isinstance(decisions, dict):
            decisions = [decisions]

        for decision in decisions:
            action = self._create_action(decision)
            if action:
                result = action.execute(self, graph)
                self.state.memory.append({
                    "layer": layer,
                    "action": result.get("action", "unknown"),
                    "result": result,
                    "text": self.state.text,
                    "label": self.state.label.item() if self.state.label is not None else None
                })

        # Fallback update logic - only trigger in the last layer
        if (layer == NUM_LAYERS - 1 and  # Only in last layer
            self.state.predicted_label is None and 
            not any(m.get("action") == "update" for m in self.state.memory) and 
            any(m.get("label") is not None for m in self.state.memory)):

            fallback_prompt = self.llm._format_fallback_label_prompt(
                self.state.text,
                self.state.memory
            )
            fallback_decision = self.llm._parse_action(
                self.llm.generate_response(fallback_prompt)
            )
            if isinstance(fallback_decision, dict) and fallback_decision.get("action_type") == "update":
                fallback_action = self._create_action(fallback_decision)
                if fallback_action:
                    fallback_result = fallback_action.execute(self, graph)
                    self.state.memory.append({
                        "layer": layer,
                        "action": "fallback_update",
                        "result": fallback_result,
                        "text": self.state.text,
                        "label": self.state.label.item() if self.state.label is not None else None
                    })
                    if DEBUG_STEP_SUMMARY:
                        print(f"\n🔄 Fallback Update | Node {self.state.node_id}")
                        print(f"  ├─ 🎯 New Label: {self.state.predicted_label}")
                        print(f"  └─ 📝 Based on {len([m for m in self.state.memory if m.get('label') is not None])} labeled examples")

        if (DEBUG_STEP_SUMMARY or DEBUG_MESSAGE_TRACE) and self.state.memory:
            last = self.state.memory[-1]
            action_type = last.get("action", "unknown")
            result = last.get("result", {})
            pred_label = self.state.predicted_label.item() if self.state.predicted_label is not None else None

            print(f"\n🧠 Agent Step | Node {self.state.node_id} | Layer {layer}")
            print(f"  ├─ 🏷️  Action: {action_type}")
            print(f"  ├─ 🎯 Predicted Label: {pred_label}")
            print(f"  ├─ 🧠 Memory size: {len(self.state.memory)}")
            print(f"  └─ 👥 Total neighbors: {len(context.get('neighbors', []))}")

        if DEBUG_MESSAGE_TRACE and self.state.memory:
            print(f"\n🔍 Message Trace | Node {self.state.node_id} | Layer {layer}")
            last = self.state.memory[-1]
            action_type = last.get("action", "unknown")
            result = last.get("result", {})

            if action_type == "retrieve":
                targets = result.get("target_nodes", [])
                results = result.get("results", {})
                print(f"  📥 Retrieved from {len(targets)} target(s):")
                for tid in targets:
                    if tid in results:
                        preview_str = self._format_preview(results[tid])
                        print(f"    ↳ Node {tid} ✅ {preview_str}")
                    else:
                        print(f"    ↳ Node {tid} ⛔ not found")
            elif action_type == "rag_query":
                print(f"  🔍 RAG Query issued: {result.get('query')} (top-k: {len(result.get('results', []))})")
            elif action_type == "broadcast":
                targets = result.get("target_nodes", [])
                message = result.get("message", None)
                print(f"  📤 Broadcasted to {len(targets)} node(s): {targets}")
                if message is not None:
                    preview = self._format_preview(message)
                    print(f"    ↳ Message: {preview}")
            elif action_type == "update":
                updated = result.get("updated_fields", [])
                print(f"  🛠️  Updated fields: {updated}")
            else:
                print("  ⚠️  No message or state updates in this step.")


    def receive_message(self, from_node: int, message: torch.Tensor) -> None:
        self.state.add_message(from_node, message, self.state.layer_count)

    def _prepare_context(self, graph: 'AgenticGraph') -> Dict[str, Any]:
        neighbors = [nid for nid in graph.get_neighbors(self.state.node_id) if nid != self.state.node_id]
        print(f"🔍 Neighbors in prepare_context: {neighbors}")
        messages = [{"from": msg["from"], "content_preview": msg["content"].mean().item(), "layer": msg["layer"]}
                    for msg in self.state.message_queue[-5:]]
        recent_memory = self.state.memory[-3:] if self.state.memory else []
        return {
            "node_id": self.state.node_id,
            "layer": self.state.layer_count,
            "text": self.state.text,
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
        action_type = decision.get("action_type", "no_op")
        
        if action_type == "retrieve":
            target_nodes = decision.get("target_nodes", [])
            info_type = decision.get("info_type", "text")
            # 确保 info_type 是支持的类型
            if info_type not in ["text", "label", "both", "memory", "all"]:
                info_type = "text"  # 默认使用 "text"
            return RetrieveAction(target_nodes, info_type)
        elif action_type == "broadcast":
            target_nodes = decision.get("target_nodes", [])
            message_data = decision.get("message", [0.0])
            if isinstance(message_data, list) and all(isinstance(x, (int, float)) for x in message_data):
                message = torch.tensor(message_data, dtype=torch.float)
            elif isinstance(message_data, (int, float)):
                message = torch.tensor([message_data], dtype=torch.float)
            else:
                message = torch.tensor([len(str(message_data))], dtype=torch.float)
            return BroadcastAction(target_nodes, message)

        elif action_type == "update":
            updates = {}
            if "predicted_label" in decision:
                label_value = decision.get("predicted_label")
                
                # Direct integer label
                if isinstance(label_value, int) and 0 <= label_value < 7:
                    updates["predicted_label"] = torch.tensor(label_value)
                    print(f"✅ Using direct integer label: {label_value}")
                
                # String label handling
                elif isinstance(label_value, str):
                    # Try parsing as integer first
                    try:
                        label_id = int(label_value)
                        if 0 <= label_id < 7:
                            updates["predicted_label"] = torch.tensor(label_id)
                            print(f"✅ Parsed label string to integer: {label_value} -> {label_id}")
                    except ValueError:
                        # Try exact match with vocabulary
                        label_id = label_vocab.get(label_value, -1)
                        if label_id != -1:
                            updates["predicted_label"] = torch.tensor(label_id)
                            print(f"✅ Mapped label string to ID: {label_value} -> {label_id}")
                        else:
                            # Try fuzzy matching
                            normalized = label_value.lower().strip()
                            if any(kw in normalized for kw in ["case", "based"]):
                                updates["predicted_label"] = torch.tensor(0)
                                print(f"✅ Fuzzy match: {label_value} -> Case_Based (0)")
                            elif any(kw in normalized for kw in ["genetic", "algorithm", "evolution"]):
                                updates["predicted_label"] = torch.tensor(1)
                                print(f"✅ Fuzzy match: {label_value} -> Genetic_Algorithms (1)")
                            elif any(kw in normalized for kw in ["neural", "network", "neuron"]):
                                updates["predicted_label"] = torch.tensor(2)
                                print(f"✅ Fuzzy match: {label_value} -> Neural_Networks (2)")
                            elif any(kw in normalized for kw in ["probabilistic", "probability", "bayes"]):
                                updates["predicted_label"] = torch.tensor(3)
                                print(f"✅ Fuzzy match: {label_value} -> Probabilistic_Methods (3)")
                            elif any(kw in normalized for kw in ["reinforcement", "reinforce"]):
                                updates["predicted_label"] = torch.tensor(4)
                                print(f"✅ Fuzzy match: {label_value} -> Reinforcement_Learning (4)")
                            elif any(kw in normalized for kw in ["rule", "rule learning"]):
                                updates["predicted_label"] = torch.tensor(5)
                                print(f"✅ Fuzzy match: {label_value} -> Rule_Learning (5)")
                            elif any(kw in normalized for kw in ["theory", "theoretical"]):
                                updates["predicted_label"] = torch.tensor(6)
                                print(f"✅ Fuzzy match: {label_value} -> Theory (6)")
                            else:
                                print(f"⚠️ Failed to map label string: {label_value}")
            
            if updates:
                return UpdateAction(updates)

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
                if all(isinstance(i, dict) for i in obj):
                    preview_list = [{k: v for k, v in item.items() if k in ("text", "predicted_label", "label")} for item in obj[:2]]
                    return str(preview_list) + ("..." if len(obj) > 2 else "")
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
