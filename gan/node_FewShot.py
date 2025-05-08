import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import os
import datetime

from gan.actions import Action, RetrieveAction, RAGAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE, NUM_LAYERS, DEBUG_FORCE_FALLBACK, RESULTS_DIR
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
        # 获取当前实验结果目录
        debug_logs_dir = os.path.join(RESULTS_DIR, "debug_logs")
        os.makedirs(debug_logs_dir, exist_ok=True)
        
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
        
        # 记录 action prompt
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(debug_logs_dir, f"node_{self.state.node_id}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"action_prompt_layer_{layer}_{timestamp}.txt"), "w", encoding="utf-8") as log_file:
            log_file.write(f"\n📤 [DEBUG] Action Prompt for Node {self.state.node_id} | Layer {layer} | {timestamp}:\n")
            log_file.write(action_prompt)
            log_file.write("\n" + "=" * 80)

        try:
            response = self.llm.generate_response(action_prompt)
            # 记录 action response
            with open(os.path.join(log_dir, f"action_response_layer_{layer}_{timestamp}.txt"), "w", encoding="utf-8") as log_file:
                log_file.write(f"🔁 [DEBUG] Action Response for Node {self.state.node_id} | Layer {layer} | {timestamp}:\n")
                log_file.write(str(response))
                log_file.write("\n" + "=" * 80)
                
            action = self.llm.parse_action(response)
            if action is None:
                fallback_prompt = self._format_simple_fallback_action_prompt(context)
                # 记录 simple fallback prompt
                with open(os.path.join(log_dir, f"simple_fallback_layer_{layer}_{timestamp}.txt"), "w", encoding="utf-8") as log_file:
                    log_file.write(f"📤 [DEBUG] Simple Fallback Prompt for Node {self.state.node_id} | Layer {layer} | {timestamp}:\n")
                    log_file.write(fallback_prompt)
                    log_file.write("\n" + "=" * 80)
                
                fallback_response = self.llm.generate_response(fallback_prompt)
                # 记录 simple fallback response
                with open(os.path.join(log_dir, f"simple_fallback_response_layer_{layer}_{timestamp}.txt"), "w", encoding="utf-8") as log_file:
                    log_file.write(f"🔁 [DEBUG] Simple Fallback Response for Node {self.state.node_id} | Layer {layer} | {timestamp}:\n")
                    log_file.write(str(fallback_response))
                    log_file.write("\n" + "=" * 80)
                    
                action = self.llm.parse_action(fallback_response)
            if isinstance(action, dict):
                action = self._create_action(action, graph)
            if action:
                result = action.execute(self, graph)
                
                # 🔧 如果是 RAGAction，补充 memory 写入
                if isinstance(action, RAGAction) and "results" in result:
                    for node_id, node_info in result["results"].items():
                        if isinstance(node_info, dict) and "text" in node_info and node_info["text"] and "label" in node_info and node_info["label"] is not None:
                            memory_entry = {
                                "layer": layer,
                                "action": "RetrieveExample",
                                "text": str(node_info["text"]),
                                "label": int(node_info["label"]),
                                "label_text": inv_label_vocab.get(int(node_info["label"]), str(node_info["label"])),
                                "source": int(node_id),
                                "source_type": "rag"
                            }
                            if not has_memory_entry(self, memory_entry):
                                self.state.memory.append(memory_entry)


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
        if layer >= NUM_LAYERS - 2 and not any(m.get("action") == "update" for m in self.state.memory):
            # 检查是否执行过retrieve
            has_retrieved = any(m.get("action") == "retrieve" for m in self.state.memory)
            # if not has_retrieved:
            if True:    ## 测试用 两次retrieve
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
            
            # 记录 fallback prompt
            with open(os.path.join(log_dir, f"fallback_layer_{layer}_{timestamp}.txt"), "w", encoding="utf-8") as log_file:
                log_file.write(f"📤 [DEBUG] Fallback Prompt for Node {self.state.node_id} | Layer {layer} | {timestamp}:\n")
                log_file.write(fallback_prompt)
                log_file.write("\n" + "=" * 80)
            
            fallback_response = self.llm.generate_response(fallback_prompt)
            fallback_decision = self.llm.parse_action(fallback_response)
            print(f"🎯 [Fallback Result]: {fallback_decision}")
            
            # 记录 fallback 响应
            with open(os.path.join(log_dir, f"fallback_response_layer_{layer}_{timestamp}.txt"), "w", encoding="utf-8") as log_file:
                log_file.write(f"🔁 [DEBUG] Fallback Response for Node {self.state.node_id} | Layer {layer} | {timestamp}:\n")
                log_file.write(str(fallback_decision))
                log_file.write("\n" + "=" * 80)

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
        prompt += "You will be given a set of labeled memory items and a new node to classify.\n"
        prompt += "Each example includes a few labeled texts as memory and a new text to classify.\n"
        prompt += "Use the memory to predict the label for the current text.\n\n"

        # === Instruction Tip ===
        prompt += "Think step-by-step. Consider which label is best supported by both the semantic content of the node text and the examples in memory.\n"
        prompt += "Do not rely on abstract or popular terms alone (like \"system\" or \"accuracy\") unless those match the label examples provided.\n\n"

        # === Example 1 ===
        prompt += "## Example 1:\n"
        prompt += "Memory:\n"
        prompt += '1. [Label_2] "Hidden Markov models for sequence modeling and pattern discovery."\n'
        prompt += '2. [Label_1] "Neural networks for text classification."\n'
        prompt += '3. [Label_2] "Bayesian models for probabilistic inference."\n'
        prompt += "Current Node Text:\n"
        prompt += '"Markov models for speech sequence alignment."\n'
        prompt += "Prediction: Label_2\n"
        prompt += "Reasoning: The text discusses Markov models and speech alignment, which closely match Label_2 examples in memory.\n\n"

        # === Example 2: 有标签冲突，靠语义推理 ===
        prompt += "## Example 2:\n"
        prompt += "Memory:\n"
        prompt += '1. [Label_2] "Hidden Markov models for biological sequence alignment."\n'
        prompt += '2. [Label_6] "Improving ensemble model selection with probabilistic voting."\n'
        prompt += '3. [Label_2] "Bayesian inference for protein sequence homology detection."\n'
        prompt += '4. [Label_6] "Boosted decision trees for structured data classification."\n'
        prompt += '5. [Label_3] "Non-reversible Markov chains for MCMC sampling."\n'
        prompt += "Current Node Text:\n"
        prompt += '"Homology detection in genetic sequences using Bayesian Markov modeling."\n'
        prompt += "Prediction: Label_2\n"
        prompt += "Reasoning: Although both Label_2 and Label_6 are well represented in memory, the current node text focuses on homology detection and Bayesian modeling, which strongly aligns with Label_2 examples related to biological sequences and probabilistic inference, rather than ensemble or structured classifiers.\n\n"

        # === 当前推理任务 ===
        prompt += "## Your Turn:\n"

        # 添加 memory label summary
        label_counts = {}
        for m in memory:
            label = m.get("label")
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts:
            prompt += "Memory Summary:\n"
            for label_id, count in sorted(label_counts.items()):
                label_str = inv_label_vocab.get(label_id, f"Label_{label_id}")
                prompt += f"- {label_str}: {count} examples\n"
            prompt += "\n"

        prompt += "Memory:\n"
        labeled_memory = [m for m in memory if m.get("label") is not None]
        for i, m in enumerate(labeled_memory[:5], 1):
            label = m.get("label")
            label_str = inv_label_vocab.get(label, f"Label_{label}")
            text = m.get("text", "").replace("\n", " ").strip()
            short_text = text[:150] + "..." if len(text) > 150 else text
            prompt += f'{i}. [{label_str}] "{short_text}"\n'

        prompt += "\nText to classify:\n"
        prompt += f'"{node_text.strip()}"\n\n'

        prompt += "Respond strictly in JSON:\n"
        prompt += '{"action_type": "update", "predicted_label": "Label_X"}' + "\n"
        label_list = ", ".join(f'\"{v}\"' for v in inv_label_vocab.values())
        prompt += f"Allowed labels: [{label_list}]\n"

        return prompt


    def _format_simple_fallback_action_prompt(self, context: Dict[str, Any]) -> str:
        return f"Based on context {context}, choose one action: retrieve, broadcast, update, or rag_query."