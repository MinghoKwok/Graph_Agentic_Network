"""
Action classes for node agents in the Graph Agentic Network
"""

import torch
from typing import Dict, List, Any, Optional, Union
from data.cora.label_vocab import inv_label_vocab  # label_id → label_name 的映射
from gan.utils import has_memory_entry


class Action:
    """Base class for all actions a node agent can take."""
    
    def execute(self, agent: 'NodeAgent', graph: 'AgenticGraph') -> Dict[str, Any]:
        """
        Execute the action and return results.
        
        Args:
            agent: The node agent executing the action
            graph: The graph environment
            
        Returns:
            Dictionary containing action results
        """
        raise NotImplementedError("Action subclasses must implement execute()")


class RetrieveAction(Action):
    """Action to retrieve information from selected neighbors."""
    
    def __init__(self, target_nodes: List[int], info_type: str = "text", aggregated_text: str = ""):
        """
        Initialize a retrieve action.

        Args:
            target_nodes: List of node IDs to retrieve from
            info_type: Type of information to retrieve ("text", "label", "both", or "all")
        """
        self.target_nodes = target_nodes
        self.info_type = info_type
        self.aggregated_text = aggregated_text
    def execute(self, agent: 'NodeAgent', graph: 'AgenticGraph') -> Dict[str, Any]:
        results = {}
        not_found = []

        for node_id in self.target_nodes:
            neighbor = graph.get_node(node_id)
            if neighbor is None:
                not_found.append(node_id)
                continue

            entry = {}
            # 获取 text
            if self.info_type in ("text", "both", "all") and neighbor.state.text:
                entry["text"] = neighbor.state.text

            # 获取 label（包括 0）
            if self.info_type in ("label", "both", "all"):
                label_tensor = neighbor.state.label if neighbor.state.label is not None else neighbor.state.predicted_label
                if label_tensor is not None:
                    try:
                        entry["label"] = label_tensor.item() if isinstance(label_tensor, torch.Tensor) else int(label_tensor)
                    except Exception:
                        entry["label"] = None

            if entry:
                results[node_id] = entry
            else:
                not_found.append(node_id)

        # ✅ 写入 memory
        for node_id, entry in results.items():
            if "text" in entry and "label" in entry and entry["label"] is not None:
                label_text = inv_label_vocab.get(entry["label"], str(entry["label"]))
                memory_entry = {
                    "layer": agent.state.layer_count,
                    "action": "RetrieveExample",
                    "text": entry["text"],
                    "label": entry["label"],
                    "label_text": label_text,
                    "source": node_id,
                    "source_type": "retrieved"
                }
                if not has_memory_entry(agent, memory_entry):
                    agent.state.memory.append(memory_entry)

        # 收集邻居 memory 中的 labeled 示例（如果有）
        for node_id, entry in results.items():
            retrieve_memory = entry.get("retrieve_memory", {})
            for cid, info in retrieve_memory.items():
                if "text" in info and "label" in info:
                    label_text = inv_label_vocab.get(info["label"], str(info["label"]))
                    memory_entry = {
                        "layer": agent.state.layer_count,
                        "action": "RetrieveExample",
                        "text": info["text"],
                        "label": info["label"],
                        "label_text": label_text,
                        "source": cid,
                        "source_type": "collected"
                    }
                    if not has_memory_entry(agent, memory_entry):
                        agent.state.memory.append(memory_entry)

        # ✅ 更新 agent.state 的 aggregated_text（如果有）
        if self.aggregated_text:
            agent.state.aggregated_text = self.aggregated_text
            print(f"🧠 [Retrieve] Node {agent.state.node_id} updated aggregated_text:\n{self.aggregated_text[:100]}...")
        else:
            print(f"⚠️ [Retrieve] Node {agent.state.node_id} missing aggregated_text from LLM. Consider fallback or debug prompt.")

        return {
            "action": "retrieve",
            "info_type": self.info_type,
            "target_nodes": self.target_nodes,
            "results": results,
            "not_found": not_found
        }

class RAGAction(Action):
    """Action to retrieve similar labeled nodes from a global knowledge base."""

    def __init__(self, query_text: str = None, top_k: int = 5):
        self.query_text = query_text
        self.top_k = top_k

    def execute(self, agent: 'NodeAgent', graph: 'AgenticGraph') -> Dict[str, Any]:
        # 如果没有提供查询文本，使用节点的文本
        query = self.query_text or agent.state.text
        
        # 执行 RAG 查询
        results = graph.rag_query(query, self.top_k)

        # ✅ 将每个结果写入 memory，格式统一为可用于 label 推理的样例
        for node_id, node_info in results.items():
            if all(key in node_info for key in ['text', 'label']):
                label = node_info['label']
                text = node_info['text']
                sim = node_info.get('similarity_score', 0.0)
                memory_entry = {
                    "layer": agent.state.layer_count,
                    "action": "RAGResult",
                    "text": text,
                    "label": label,
                    "label_text": inv_label_vocab.get(label, str(label)),
                    "source": node_id,
                    "source_type": "rag",
                    "similarity_score": sim
                }
                if not has_memory_entry(agent, memory_entry):
                    agent.state.memory.append(memory_entry)

        return {
            "action": "rag_query",
            "query": query,
            "results": results
        }

class BroadcastAction(Action):
    """Broadcast message to target nodes."""
    
    def __init__(self, target_nodes: List[int], message: torch.Tensor):
        """
        Initialize a broadcast action.
        
        Args:
            target_nodes: List of node IDs to broadcast to
            message: The message tensor to broadcast
        """
        self.target_nodes = target_nodes
        self.message = message
        
    def execute(self, agent: 'NodeAgent', graph: 'AgenticGraph') -> Dict[str, Any]:
        """
        Execute broadcast action to target nodes.
        
        Args:
            agent: The node agent executing the action
            graph: The graph containing all nodes
            
        Returns:
            Dictionary containing action results
        """
        from data.cora.label_vocab import inv_label_vocab  # 本地导入，防止循环

        if not self.target_nodes:
            return {"action": "no_op", "message": None, "target_nodes": []}

        print(f"📤 [Action: Broadcast] Node {agent.state.node_id} → {self.target_nodes}")

        # Step 1: 构造消息
        message_payload = None
        label_tensor = agent.state.predicted_label or agent.state.label
        if label_tensor is not None:
            message_payload = {
                "text": agent.state.text,
                "predicted_label": label_tensor.item()
            }
        elif agent.state.memory:
            labeled_examples = [
                m for m in agent.state.memory if m.get("label") is not None
            ]
            if labeled_examples:
                message_payload = labeled_examples

        if message_payload is None:
            return {"action": "no_op", "message": None, "target_nodes": []}

        # Step 2: 发送给每个目标节点
        for target_id in self.target_nodes:
            target_agent = graph.get_node(target_id)
            if not target_agent:
                print(f"    ⛔ Node {target_id} not found in graph")
                continue

            # Step 2.1: 防止重复写入
            already_seen = False
            for m in target_agent.state.memory:
                if m.get("action") == "broadcast":
                    if m.get("result", {}).get("message") == message_payload:
                        already_seen = True
                        break
            if already_seen:
                continue

            # Step 2.2: 写入原始广播信息
            memory_entry = {
                "action": "broadcast",
                "result": {
                    "message": message_payload,
                    "source": agent.state.node_id
                }
            }
            if not has_memory_entry(target_agent, memory_entry):
                target_agent.state.memory.append(memory_entry)

            # Step 2.3: 如果 message 是带标签的 dict，则额外写入可用 labeled 示例
            if isinstance(message_payload, dict) and "predicted_label" in message_payload:
                label_id = message_payload["predicted_label"]
                text = message_payload["text"]
                label_text = inv_label_vocab.get(label_id, str(label_id))
                memory_entry = {
                    "layer": agent.state.layer_count,
                    "action": "BroadcastLabel",
                    "text": text,
                    "label": label_id,
                    "label_text": label_text,
                    "source": agent.state.node_id,
                    "source_type": "broadcast"
                }
                if not has_memory_entry(target_agent, memory_entry):
                    target_agent.state.memory.append(memory_entry)

        return {
            "action": "broadcast",
            "message": message_payload,
            "target_nodes": self.target_nodes
        }


class UpdateAction(Action):
    """Action to update the node's own state."""
    
    def __init__(self, updates: Dict[str, Union[torch.Tensor, Any]]):
        """
        Initialize an update action.
        
        Args:
            updates: Dictionary of state updates with keys like "features", "hidden_state", "label"
        """
        self.updates = updates
        
    def execute(self, agent: 'NodeAgent', graph: 'AgenticGraph') -> Dict[str, Any]:
        """
        Execute the update action.
        
        Args:
            agent: The node agent executing the action
            graph: The graph environment
            
        Returns:
            Dictionary containing update results
        """
        updated_fields = []
        
        for key, value in self.updates.items():
            if key == "hidden_state" and isinstance(value, torch.Tensor):
                agent.state.hidden_state = value
                updated_fields.append("hidden_state")
            elif key == "predicted_label" and isinstance(value, (torch.Tensor, int)):
                if isinstance(value, int):
                    agent.state.predicted_label = torch.tensor(value)
                else:
                    agent.state.predicted_label = value
                updated_fields.append("predicted_label")
            elif key == "text" and isinstance(value, str):
                agent.state.text = value
                updated_fields.append("text")
        
        return {
            "action": "update",
            "updated_fields": updated_fields
        }


class NoOpAction(Action):
    """Action that does nothing."""
    
    def execute(self, agent: 'NodeAgent', graph: 'AgenticGraph') -> Dict[str, Any]:
        """
        Execute the no-op action.
        
        Args:
            agent: The node agent executing the action
            graph: The graph environment
            
        Returns:
            Empty result dictionary
        """
        return {
            "action": "no_op"
        }
