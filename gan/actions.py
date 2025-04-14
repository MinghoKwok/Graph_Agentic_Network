"""
Action classes for node agents in the Graph Agentic Network
"""

import torch
from typing import Dict, List, Any, Optional, Union
from data.cora.label_vocab import inv_label_vocab  # label_id → label_name 的映射


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
        """
        Execute the retrieve action.

        Args:
            agent: The node agent executing the action
            graph: The graph environment

        Returns:
            Dictionary containing retrieved information
        """
        results = {}
        not_found = []

        for node_id in self.target_nodes:
            neighbor = graph.get_node(node_id)
            if neighbor is None:
                not_found.append(node_id)
                continue

            entry = {}
            # 获取 text 和 label
            if self.info_type in ("text", "both", "all") and neighbor.state.text:
                entry["text"] = neighbor.state.text
            if self.info_type in ("label", "both", "all"):
                entry["label"] = neighbor.state.label.item() if neighbor.state.label is not None else None

            if entry:
                results[node_id] = entry
            else:
                not_found.append(node_id)

        # ✅ 将 labeled entries 写入当前节点 memory，加入 label_text（自然语言标签）
        for node_id, entry in results.items():
            if "text" in entry and "label" in entry and entry["label"] is not None:
                label_text = inv_label_vocab.get(entry["label"], str(entry["label"]))
                agent.state.memory.append({
                    "layer": agent.state.layer_count,  # 当前层数
                    "action": "RetrieveExample",
                    "text": entry["text"],
                    "label": entry["label"],           # 数值标签
                    "label_text": label_text,          # 文本标签
                    "source": node_id,
                    "source_type": "retrieved"
                })
        # 将邻居 memory 中的 labeled 示例也写入 memory
        for node_id, entry in results.items():
            retrieve_memory = entry.get("retrieve_memory", {})
            for cid, info in retrieve_memory.items():
                if "text" in info and "label" in info:
                    label_text = inv_label_vocab.get(info["label"], str(info["label"]))
                    agent.state.memory.append({
                        "layer": agent.state.layer_count,
                        "action": "RetrieveExample",
                        "text": info["text"],
                        "label": info["label"],
                        "label_text": label_text,
                        "source": cid,
                        "source_type": "collected"
                    })


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
        
        # 将结果写入节点的记忆中
        for node_id, node_info in results.items():
            # 确保节点信息包含必要的字段
            if all(key in node_info for key in ['text', 'label']):
                agent.state.memory.append({
                    'action': 'rag_query',
                    'result': {
                        'node_id': node_id,
                        'text': node_info['text'],
                        'label': node_info['label'],
                        'similarity_score': node_info.get('similarity_score', 0.0)
                    }
                })
        
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
        if not self.target_nodes:
            return {"action": "no_op", "message": None, "target_nodes": []}
            
        # ✅ 在 BroadcastAction.execute 中添加 debug log，确认是否正确发送消息
        print(f"📤 [Broadcast] Node {agent.state.node_id} → {self.target_nodes}")
        for nid in self.target_nodes:
            if graph.has_node(nid):
                print(f"    ↳ ✅ Sending to Node {nid}")
            else:
                print(f"    ↳ ⛔ Node {nid} not found in graph")
            
        # Prepare message payload based on available information
        message_payload = None
        
        # Case 1: Broadcast predicted label if available
        if agent.state.predicted_label is not None:
            message_payload = {
                "text": agent.state.text,
                "predicted_label": agent.state.predicted_label.item()
            }
        # Case 2: Broadcast labeled examples from memory if available
        elif agent.state.memory:
            labeled_examples = [
                m for m in agent.state.memory 
                if m.get("label") is not None
            ]
            if labeled_examples:
                message_payload = labeled_examples
                
        # If no valid message payload, return no_op
        if message_payload is None:
            return {"action": "no_op", "message": None, "target_nodes": []}
            
        # Send message to target nodes
        for target_id in self.target_nodes:
            target_agent = graph.get_node(target_id)
            if target_agent:
                # Check for duplicate messages in target's memory
                is_duplicate = False
                for m in target_agent.state.memory:
                    if isinstance(m, dict) and m.get("action") == "broadcast":
                        if (isinstance(message_payload, dict) and 
                            m.get("result", {}).get("message") == message_payload):
                            is_duplicate = True
                            break
                        elif (isinstance(message_payload, list) and 
                              m.get("result", {}).get("message") == message_payload):
                            is_duplicate = True
                            break
                            
                if not is_duplicate:
                    target_agent.state.memory.append({
                        "action": "broadcast",
                        "result": {
                            "message": message_payload,
                            "source": agent.state.node_id
                        }
                    })
                    
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
