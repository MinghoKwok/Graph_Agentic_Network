"""
Node agent implementation for the Graph Agentic Network
"""

import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from gan.actions import Action, RetrieveAction, RAGAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE, NUM_LAYERS, DEBUG_FORCE_FALLBACK  # 加入 NUM_LAYERS 以判断是否为最后一层
from data.cora.label_vocab import label_vocab  # 自定义标签映射
from gan.utils import has_memory_entry

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

    def step(self, graph: 'AgenticGraph', layer: int):
        context = {
            "node_id": self.state.node_id,
            "text": self.state.text,
            "label": self.state.label.item() if self.state.label is not None else None,
            "layer": layer,
            "memory": self.state.memory,
            "neighbors": graph.get_neighbors(self.state.node_id),
            "total_neighbors": len(graph.get_neighbors(self.state.node_id))
        }

        self.skip_update = hasattr(graph, "train_idx") and self.state.node_id in graph.train_idx

        action_list = []
        try:
            response = self.llm.generate_response(self.llm._format_action_prompt(context))
            parsed = self.llm.parse_action(response)
            if isinstance(parsed, dict):
                action_list = [parsed]
            elif isinstance(parsed, list):
                action_list = parsed
            else:
                action_list = [NoOpAction()]
        except Exception as e:
            print(f"⚠️ Error in agent step: {e}")
            fallback_prompt = self.llm._format_fallback_label_prompt(self.state.text, self.state.memory)
            fallback_response = self.llm.generate_response(fallback_prompt)
            fallback_decision = self.llm.parse_action(fallback_response)
            if isinstance(fallback_decision, dict):
                action_list = [fallback_decision]
            elif isinstance(fallback_decision, list):
                action_list = fallback_decision
            else:
                action_list = [NoOpAction()]

        print(f"\n📋 Multi-Action Plan | Node {self.state.node_id} | Layer {layer}")
        for idx, d in enumerate(action_list):
            print(f"  {idx+1}. {d}")

        for decision in action_list:
            action = self._create_action(decision, graph)
            if action:
                action_type = decision.get("action_type") if isinstance(decision, dict) else action.__class__.__name__
                result = action.execute(self, graph)
                print(f"✅ Executed {action_type} with result: {result}")
                if not has_memory_entry(self, result):
                    self.state.memory.append({
                        "layer": layer,
                        "action": result.get("action", action_type),
                        "result": result,
                        "text": self.state.text,
                        "label": self.state.label.item() if self.state.label is not None else None
                    })

        # fallback update 仅在最后一层触发
        if (self.skip_update == False and layer == NUM_LAYERS - 1 and (DEBUG_FORCE_FALLBACK or (
            self.state.predicted_label is None and 
            not any(m.get("action") == "update" for m in self.state.memory) and 
            any(m.get("label") is not None for m in self.state.memory)))):

            fallback_prompt = self.llm._format_fallback_label_prompt(self.state.text, self.state.memory)
            print(f"\n📦 [Fallback Prompt for Node {self.state.node_id}]:\n{fallback_prompt}")
            fallback_response = self.llm.generate_response(fallback_prompt)
            fallback_decision = self.llm.parse_action(fallback_response)
            fallback_actions = fallback_decision if isinstance(fallback_decision, list) else [fallback_decision]

            for decision in fallback_actions:
                if isinstance(decision, dict) and decision.get("action_type") == "update":
                    fallback_action = self._create_action(decision, graph)
                    if fallback_action:
                        fallback_result = fallback_action.execute(self, graph)
                        if not has_memory_entry(self, fallback_result):
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


    def receive_message(self, from_node: int, message: torch.Tensor) -> None:
        # ✅ 在 receive_message 或 receive_broadcast 中也加入一行确认接收
        print(f"📨 Node {self.state.node_id} received message from Node {from_node}")
        self.state.add_message(from_node, message, self.state.layer_count)

    def _prepare_context(self, graph: 'AgenticGraph') -> Dict[str, Any]:
        neighbors = [nid for nid in graph.get_neighbors(self.state.node_id) if nid != self.state.node_id]
        print(f"🔍 Neighbors in prepare_context: {neighbors}")
        
        updated_neighbors = [
            nid for nid in neighbors
            if graph.get_node(nid).state.predicted_label is not None
        ]
        print(f"📊 Updated neighbors (with predicted labels): {updated_neighbors}")

        # ✅ 添加 labeled_neighbors
        labeled_neighbors = [
            nid for nid in neighbors
            if graph.get_node(nid).state.label is not None
        ]
        print(f"🏷️ Labeled neighbors: {labeled_neighbors}")

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
            "updated_neighbors": updated_neighbors,
            "labeled_neighbors": labeled_neighbors,  # ✅ 新增字段
            "messages": messages,
            "total_messages": len(self.state.message_queue),
            "memory": recent_memory,
            "total_memory": len(self.state.memory),
            "retrieved_data": self.retrieved_data
        }

    def _create_action(self, decision: Dict[str, Any], graph: 'AgenticGraph') -> Optional[Action]:
        if isinstance(decision, dict):
            action_type = decision.get("action_type", "no_op")
        elif isinstance(decision, Action):
            return decision  # 已经是构造好的 Action，直接返回
        else:
            print(f"⚠️ Unsupported decision type: {type(decision)}. Fallback to NoOp.")
            return NoOpAction()
        
        # 如果一个节点本身已经有 label，且是 training node，就不让它进行 update（因为我们不需要 predicted_label）
        if getattr(self, "skip_update", False) and action_type == "update":
            print(f"🛑 Node {self.state.node_id} is in training set. Skipping update.")
            return NoOpAction()


        if action_type == "retrieve":
            target_nodes = decision.get("target_nodes", [])
            info_type = decision.get("info_type", "text")
            # 确保 info_type 是支持的类型
            if info_type not in ["text", "label", "both", "memory", "all"]:
                info_type = "text"  # 默认使用 "text"
            return RetrieveAction(target_nodes, info_type)
        
        elif action_type == "rag_query":
            query = decision.get("query", str(self.state.node_id))  # 默认使用节点ID作为查询
            top_k = decision.get("top_k", 5)  # 默认获取5个相似节点
            query = str(self.state.node_id)  # 👈 强制使用节点自身的 ID 作为 query
            return RAGAction(query, top_k)

        elif action_type == "broadcast":
            target_nodes = decision.get("target_nodes")
            if not target_nodes:
                target_nodes = graph.get_neighbors(self.state.node_id)
                print(f"⚠️ Broadcast targets missing — fallback to all neighbors: {target_nodes}")

            # ✅ fallback message logic
            message_data = decision.get("message", None)

            if message_data is None:
                # fallback to predicted_label + text
                plabel = self.state.predicted_label.item() if self.state.predicted_label is not None else "unknown"
                text = self.state.text[:60] + "..." if len(self.state.text) > 60 else self.state.text
                fallback_message = f"[Label: {plabel}] {text}"
                print(f"⚠️ Broadcast message missing — fallback to: {fallback_message}")
                message_data = fallback_message

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
            else:
                print(f"⚠️ No valid predicted_label found in update decision: {decision}")

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
