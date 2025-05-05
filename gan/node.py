import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from gan.actions import Action, RetrieveAction, RAGAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE, NUM_LAYERS, DEBUG_FORCE_FALLBACK
from data.cora.label_vocab import label_vocab, inv_label_vocab
from gan.utils import has_memory_entry

@dataclass
class NodeState:
    node_id: int
    text: str
    label: Optional[torch.Tensor] = None
    predicted_label: Optional[torch.Tensor] = None
    message_queue: List[Dict[str, Any]] = field(default_factory=list)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    layer_count: int = 0

    def add_message(self, from_node: int, message: torch.Tensor, layer: int):
        self.message_queue.append({"from": from_node, "content": message, "layer": layer})

    def clear_messages(self):
        self.message_queue = []

    def increment_layer(self):
        self.layer_count += 1

class NodeAgent:
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

        action_prompt = self.llm._format_action_prompt(context)

        try:
            response = self.llm.generate_response(action_prompt)
            action = self.llm.parse_action(response)
            if action is None:
                fallback_prompt = self._format_simple_fallback_action_prompt(context)
                fallback_response = self.llm.generate_response(fallback_prompt)
                action = self.llm.parse_action(fallback_response)
            if isinstance(action, dict):
                action = self._create_action(action, graph)
            if action:
                result = action.execute(self, graph)
                if isinstance(result, dict):
                    if result.get("action_type") == "retrieve" and "results" in result:
                        # 处理retrieve结果
                        for node_id, node_info in result["results"].items():
                            if isinstance(node_info, dict) and "text" in node_info and node_info["text"]:
                                memory_entry = {
                                    "layer": layer,
                                    "action": "RetrieveExample",
                                    "text": str(node_info['text']),
                                    "label": int(node_info['label']) if 'label' in node_info and node_info['label'] is not None else None,
                                    "label_text": inv_label_vocab.get(int(node_info['label']), str(node_info['label'])) if 'label' in node_info and node_info['label'] is not None else None,
                                    "source": int(node_id),
                                    "source_type": "retrieved"
                                }
                                if not has_memory_entry(self, memory_entry):
                                    self.state.memory.append(memory_entry)
                    elif result.get("action_type") == "update":
                        # 处理update结果
                        memory_entry = {
                            "layer": layer,
                            "action": "update",
                            "text": self.state.text,
                            "label": self.state.predicted_label.item() if self.state.predicted_label is not None else None,
                            "label_text": inv_label_vocab.get(int(self.state.predicted_label.item()), str(self.state.predicted_label.item())) if self.state.predicted_label is not None else None,
                            "source": self.state.node_id,
                            "source_type": "self"
                        }
                        if not has_memory_entry(self, memory_entry):
                            self.state.memory.append(memory_entry)
                    else:
                        # 其他action的结果
                        memory_entry = {
                            "layer": layer,
                            "action": action.__class__.__name__,
                            "result": result
                        }
                        if not has_memory_entry(self, memory_entry):
                            self.state.memory.append(memory_entry)
        except Exception as e:
            print(f"⚠️ Error in agent step: {e}")

        # === Fallback update logic: only trigger at the last layer ===
        # Trigger fallback update if:
        # - in the last layer
        # - no update action occurred
        if layer == NUM_LAYERS - 1 and not any(m.get("action") == "update" for m in self.state.memory):
            # 检查是否执行过retrieve
            has_retrieved = any(m.get("action") == "retrieve" for m in self.state.memory)
            if not has_retrieved:
                # 先执行一次retrieve
                print(f"\n🔄 [Fallback Retrieve] Node {self.state.node_id} | Layer {layer}")
                retrieve_action = RetrieveAction(
                    target_nodes=graph.get_neighbors(self.state.node_id),
                    info_type="all"
                )
                retrieve_result = retrieve_action.execute(self, graph)
                
                # 处理retrieve结果
                if isinstance(retrieve_result, dict) and "results" in retrieve_result:
                    for node_id, node_info in retrieve_result["results"].items():
                        if isinstance(node_info, dict) and "text" in node_info and node_info["text"]:
                            memory_entry = {
                                "layer": layer,
                                "action": "RetrieveExample",
                                "text": str(node_info['text']),
                                "label": int(node_info['label']) if 'label' in node_info and node_info['label'] is not None else None,
                                "label_text": inv_label_vocab.get(int(node_info['label']), str(node_info['label'])) if 'label' in node_info and node_info['label'] is not None else None,
                                "source": int(node_id),
                                "source_type": "retrieved"
                            }
                            if not has_memory_entry(self, memory_entry):
                                self.state.memory.append(memory_entry)

            # 构造 fallback prompt 并推理
            fallback_prompt = self._format_fallback_label_prompt(self.state.text, self.state.memory)
            print(f"\n📦 [Fallback Prompt for Node {self.state.node_id}]:\n{fallback_prompt}")
            fallback_response = self.llm.generate_response(fallback_prompt)
            fallback_decision = self.llm.parse_action(fallback_response)
            print(f"🎯 [Fallback Result]: {fallback_decision}")

            if isinstance(fallback_decision, dict) and fallback_decision.get("action_type") == "update":
                fallback_action = self._create_action(fallback_decision, graph)
                if fallback_action:
                    fallback_result = fallback_action.execute(self, graph)
                    if not has_memory_entry(self, fallback_result):
                        self.state.memory.append({
                            "layer": layer,
                            "action": "update",
                            "text": self.state.text,
                            "label": self.state.predicted_label.item() if self.state.predicted_label is not None else None,
                            "label_text": inv_label_vocab.get(int(self.state.predicted_label.item()), str(self.state.predicted_label.item())) if self.state.predicted_label is not None else None,
                            "source": self.state.node_id,
                            "source_type": "self"
                        })
                    if DEBUG_STEP_SUMMARY:
                        print(f"\n🔄 Fallback Update | Node {self.state.node_id}")
                        print(f"  ├─ 🎯 New Label: {self.state.predicted_label}")
                        print(f"  └─ 📝 Based on {len([m for m in self.state.memory if m.get('label') is not None])} labeled examples")
            else:
                print(f"⚠️ Fallback decision did not yield a valid update action. Skipping fallback update.")

    def _create_action(self, decision: Dict[str, Any], graph: 'AgenticGraph') -> Optional[Action]:
        if isinstance(decision, dict):
            action_type = decision.get("action_type", "no_op")
        elif isinstance(decision, Action):
            return decision
        else:
            return NoOpAction()

        if getattr(self, "skip_update", False) and action_type == "update":
            return NoOpAction()

        if action_type == "retrieve":
            return RetrieveAction(decision.get("target_nodes", []), decision.get("info_type", "text"))
        elif action_type == "rag_query":
            return RAGAction(self.state.node_id, decision.get("top_k", 5))
        elif action_type == "broadcast":
            return BroadcastAction(decision.get("target_nodes", []), torch.tensor([len(str(decision.get("message", "")))], dtype=torch.float))
        elif action_type == "update":
            updates = {}
            label_value = decision.get("predicted_label")
            if isinstance(label_value, int) and 0 <= label_value < 7:
                updates["predicted_label"] = torch.tensor(label_value)
            elif isinstance(label_value, str):
                label_id = label_vocab.get(label_value, -1)
                if label_id != -1:
                    updates["predicted_label"] = torch.tensor(label_id)
            if updates:
                return UpdateAction(updates)
        return NoOpAction()

    def _format_fallback_label_prompt(self, node_text: str, memory: List[Dict[str, Any]]) -> str:
        prompt = "You are a label prediction agent.\n\n"
        prompt += f"Text to classify:\n\"{node_text}\"\n\n"
        if memory:
            prompt += "Memory items:\n"
            for m in memory:
                label = m.get("label", None)
                text = m.get("text", "")
                if label is not None:
                    label_str = inv_label_vocab.get(label, f"Label_{label}")
                    short_text = text[:60] + "..." if len(text) > 60 else text
                    prompt += f"- \"{short_text}\" — label: {label_str}\n"
        else:
            prompt += "Memory items: (No memory available)\n"

        # prompt += "\nOfficial Label Definitions:\n"
#         prompt += """
# [label=Label_6]
# - Development of abstract models.
# [label=Neural_Networks]
# - Multi-layered network models.
# [label=Case_Based]
# - Reasoning by past examples.
# [label=Genetic_Algorithms]
# - Evolutionary optimization methods.
# [label=Probabilistic_Methods]
# - Probability and Bayesian methods.
# [label=Reinforcement_Learning]
# - Learning from rewards and actions.
# [label=Rule_Learning]
# - Extracting if-then symbolic rules.
# """
        prompt += "\nRespond strictly in JSON format:\n{\"action_type\": \"update\", \"predicted_label\": \"label_string\"}\n"
        label_list = ", ".join(f'\"{v}\"' for v in inv_label_vocab.values())
        prompt += f"Allowed labels: [{label_list}]\n"
        return prompt

    def _format_simple_fallback_action_prompt(self, context: Dict[str, Any]) -> str:
        return f"Based on context {context}, choose one action: retrieve, broadcast, update, or rag_query."