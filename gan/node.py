"""
Node agent implementation for the Graph Agentic Network
"""

import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from gan.actions import Action, RetrieveAction, RAGAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE, NUM_LAYERS, DEBUG_FORCE_FALLBACK  # 加入 NUM_LAYERS 以判断是否为最后一层
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

    def step(self, graph: 'AgenticGraph', layer: int):
        """Execute one step of the agent's decision-making process."""
        # 准备上下文信息
        context = {
            "node_id": self.state.node_id,
            "text": self.state.text,
            "label": self.state.label.item() if self.state.label is not None else None,
            "layer": layer,
            "memory": self.state.memory,
            "neighbors": graph.get_neighbors(self.state.node_id),
            "total_neighbors": len(graph.get_neighbors(self.state.node_id))
        }
        
        # 获取动作提示
        action_prompt = self.llm._format_action_prompt(context, graph)
        
        try:
            # 生成响应并解析动作
            response = self.llm.generate_response(action_prompt)
            action = self.llm.parse_action(response)
            
            # 如果解析失败，使用备用决策
            if action is None:
                print(f"⚠️ Failed to parse action from response: {response}")
                fallback_prompt = f"""Based on the following context, choose the most appropriate action:
Context: {context}
Available actions: retrieve, broadcast, update, rag_query
Choose one action and provide parameters."""
                fallback_decision = self.llm.parse_action(self.llm.generate_response(fallback_prompt))
                if fallback_decision is None:
                    print("⚠️ Fallback decision also failed. Using default update action.")
                    action = UpdateAction()
                else:
                    action = fallback_decision
            
            # 执行动作
            if isinstance(action, dict):
                action = self._create_action(action)
            if action:
                result = action.execute(self, graph)
            
            # 更新记忆
            self.state.memory.append({
                "layer": layer,
                "action": action.__class__.__name__,
                "result": result
            })
            
        except Exception as e:
            print(f"⚠️ Error in agent step: {e}")
            # 使用 fallback update 动作结构，确保含有合法参数
            fallback_prompt = self.llm._format_fallback_label_prompt(self.state.text, self.state.memory)
            print(f"\n📦 [Exception Fallback Prompt for Node {self.state.node_id}]:\n{fallback_prompt}")
            fallback_response = self.llm.generate_response(fallback_prompt)
            fallback_decision = self.llm.parse_action(fallback_response)
            print(f"🎯 [Exception Fallback Result]: {fallback_decision}")

            if isinstance(fallback_decision, dict) and fallback_decision.get("action_type") == "update":
                action = self._create_action(fallback_decision)
            else:
                action = NoOpAction()  # 最坏情况也不要直接 new UpdateAction()

            if isinstance(action, dict):
                action = self._create_action(action)
            if action:
                result = action.execute(self, graph)

            self.state.memory.append({
                "layer": layer,
                "action": action.__class__.__name__,
                "result": result,
                "error": str(e)
            })


        # ✅ 插入在 step() 函数最开始，打印每个节点当前计划的完整 action 列表
        print(f"\n📋 Multi-Action Plan | Node {self.state.node_id} | Layer {layer}")

        # ✅ 统一包成 list，无论是 dict 还是 Action 实例
        if isinstance(action, dict):
            action_list = [action]
        elif isinstance(action, Action):
            action_list = [action]
        elif isinstance(action, list):
            action_list = action
        else:
            action_list = [NoOpAction()]  # fallback

        for idx, d in enumerate(action_list):
            print(f"  {idx+1}. {d}")


        # Ensure decisions is a list
        # Normalize the action output to a list to support multiple sequential actions per node step.
        # This enables LLMs to plan a sequence like: [retrieve → update → broadcast]

        for decision in action_list:
            action = self._create_action(decision)
            if action:
                action_type = decision.get("action_type") if isinstance(decision, dict) else action.__class__.__name__
                result = action.execute(self, graph)
                print(f"✅ Executed {action_type} with result: {result}")
                self.state.memory.append({
                    "layer": layer,
                    "action": result.get("action", "unknown"),
                    "result": result,
                    "text": self.state.text,
                    "label": self.state.label.item() if self.state.label is not None else None
                })

        # === Fallback update logic: only trigger at the last layer ===
        # Trigger fallback update only if:
        # - in the last layer,
        # - no predicted label yet,
        # - no prior update action occurred,
        # - but memory contains labeled examples.
        if (layer == NUM_LAYERS - 1 and (DEBUG_FORCE_FALLBACK or (
            self.state.predicted_label is None and 
            not any(m.get("action") == "update" for m in self.state.memory) and 
            any(m.get("label") is not None for m in self.state.memory)))):

            # 构造 fallback prompt 并推理
            fallback_prompt = self.llm._format_fallback_label_prompt(self.state.text, self.state.memory)
            print(f"\n📦 [Fallback Prompt for Node {self.state.node_id}]:\n{fallback_prompt}")
            fallback_response = self.llm.generate_response(fallback_prompt)
            fallback_decision = self.llm.parse_action(fallback_response)
            print(f"🎯 [Fallback Result]: {fallback_decision}")

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
            else:
                print(f"⚠️ Fallback decision did not yield a valid update action. Skipping fallback update.")

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
            # Show agent's most recent action result for debugging and traceability.
            # Useful for layer-wise inspection of node behavior.


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
        # ✅ 在 receive_message 或 receive_broadcast 中也加入一行确认接收
        print(f"📨 Node {self.state.node_id} received message from Node {from_node}")
        self.state.add_message(from_node, message, self.state.layer_count)

    def _prepare_context(self, graph: 'AgenticGraph') -> Dict[str, Any]:
        neighbors = [nid for nid in graph.get_neighbors(self.state.node_id) if nid != self.state.node_id]
        print(f"🔍 Neighbors in prepare_context: {neighbors}")
        
        # 找出已 update 的邻居（即有 predicted_label 的邻居）
        updated_neighbors = [
            nid for nid in neighbors
            if graph.get_node(nid).state.predicted_label is not None
        ]
        print(f"📊 Updated neighbors (with predicted labels): {updated_neighbors}")
        
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
            "updated_neighbors": updated_neighbors,  # 添加已更新邻居列表
            "messages": messages,
            "total_messages": len(self.state.message_queue),
            "memory": recent_memory,
            "total_memory": len(self.state.memory),
            "retrieved_data": self.retrieved_data
        }

    def _create_action(self, decision: Dict[str, Any]) -> Optional[Action]:
        if isinstance(decision, dict):
            action_type = decision.get("action_type", "no_op")
        elif isinstance(decision, Action):
            return decision  # 已经是构造好的 Action，直接返回
        else:
            print(f"⚠️ Unsupported decision type: {type(decision)}. Fallback to NoOp.")
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
            target_nodes = decision.get("target_nodes", [])

            # ✅ fallback message logic
            message_data = decision.get("message", None)

            if message_data is None:
                # fallback to predicted_label + text
                # If LLM does not provide a broadcast message, fallback to a default message combining predicted_label + text.
                # This ensures all broadcast actions remain valid and meaningful for downstream nodes.
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
