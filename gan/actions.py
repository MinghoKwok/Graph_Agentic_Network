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
    
    def __init__(self, target_nodes: List[int], info_type: str = "text"):
        """
        Initialize a retrieve action.

        Args:
            target_nodes: List of node IDs to retrieve from
            info_type: Type of information to retrieve ("text", "label", "both", or "all")
        """
        self.target_nodes = target_nodes
        self.info_type = info_type

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
                        if isinstance(label_tensor, torch.Tensor):
                            entry["label"] = label_tensor.item()
                        elif isinstance(label_tensor, (int, float)):
                            entry["label"] = int(label_tensor)
                        else:
                            entry["label"] = None
                    except Exception:
                        entry["label"] = None

            if entry:
                results[node_id] = entry

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
        # 获取节点ID作为查询
        query_id = agent.state.node_id
        print(f"[DEBUG] RAGAction: 执行节点 {query_id} 的RAG查询, top_k={self.top_k}")
        
        # 检查向量索引是否可用
        if not hasattr(graph, '_node_features_index') or graph._node_features_index is None:
            print(f"[DEBUG] ❌ RAGAction: 图没有可用的向量索引")
            return {
                "action": "rag_query",
                "query_id": query_id,
                "error": "Vector index not available",
                "results": {}
            }
        
        # 检查当前节点是否有embedding向量
        if not hasattr(graph, '_node_id_to_query_vector'):
            print(f"[DEBUG] ❌ RAGAction: 图没有_node_id_to_query_vector属性")
            return {
                "action": "rag_query",
                "query_id": query_id,
                "error": "No embedding dictionary available",
                "results": {}
            }
        
        node_embeddings = graph._node_id_to_query_vector
        if query_id not in node_embeddings:
            print(f"[DEBUG] ❌ RAGAction: 节点 {query_id} 在embedding字典中不存在")
            return {
                "action": "rag_query",
                "query_id": query_id,
                "error": f"Node {query_id} has no embedding",
                "results": {}
            }
        
        # 执行 RAG 查询
        results = graph.rag_query(query_id, self.top_k)
        
        # 返回结果，不在这里写入memory，让NodeAgent统一处理
        return {
            "action": "rag_query",
            "query_id": query_id,
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
