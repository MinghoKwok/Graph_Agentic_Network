import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time
import signal
from contextlib import contextmanager

from gan.actions import Action, RetrieveAction, RAGAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE, NUM_LAYERS, DEBUG_FORCE_FALLBACK
from data.cora.label_vocab import label_vocab, inv_label_vocab
from gan.utils import has_memory_entry

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Action execution timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

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
        # 初始化动作执行历史记录
        self.action_history = []
        # 初始化动作执行统计信息
        self.action_stats = {
            "success": 0,
            "failure": 0,
            "timeout": 0,
            "by_type": {}
        }

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
            actions = self.llm.parse_action(response)

            if not actions:
                fallback_prompt = self._format_simple_fallback_action_prompt(context)
                fallback_response = self.llm.generate_response(fallback_prompt)
                actions = self.llm.parse_action(fallback_response)

            if isinstance(actions, dict):
                actions = [actions]

            if isinstance(actions, list):
                for action_dict in actions:
                    if not isinstance(action_dict, dict):
                        print(f"⚠️ Invalid action format: {action_dict}")
                        self._update_action_stats("failure")
                        continue

                    action = self._create_action(action_dict, graph)
                    if not action:
                        print(f"⚠️ Failed to create action from: {action_dict}")
                        self._update_action_stats("failure")
                        continue

                    try:
                        with timeout(5):  # 5秒超时
                            result = action.execute(self, graph)
                            if result is None:
                                print(f"⚠️ Action execution returned None: {action.__class__.__name__}")
                                self._update_action_stats("failure")
                                continue

                            # 记录动作执行历史
                            self._record_action_history(layer, action, result)
                            # 更新统计信息
                            self._update_action_stats("success", action.__class__.__name__)

                            if isinstance(result, dict):
                                if result.get("action_type") == "retrieve" and "results" in result:
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
                                    memory_entry = {
                                        "layer": layer,
                                        "action": action.__class__.__name__,
                                        "result": result
                                    }
                                    if not has_memory_entry(self, memory_entry):
                                        self.state.memory.append(memory_entry)
                            else:
                                memory_entry = {
                                    "layer": layer,
                                    "action": action.__class__.__name__,
                                    "result": str(result)
                                }
                                if not has_memory_entry(self, memory_entry):
                                    self.state.memory.append(memory_entry)
                    except TimeoutError:
                        print(f"⚠️ Action execution timed out: {action.__class__.__name__}")
                        self._update_action_stats("timeout")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error executing action {action.__class__.__name__}: {e}")
                        self._update_action_stats("failure")
                        continue
        except Exception as e:
            print(f"⚠️ Error in agent step: {e}")

        # === Fallback update logic: only trigger at the last layer ===
        if layer == NUM_LAYERS - 1 and not any(m.get("action") == "update" for m in self.state.memory):
            has_retrieved = any(m.get("action") == "retrieve" for m in self.state.memory)
            if not has_retrieved:
                print(f"\n🔄 [Fallback Retrieve] Node {self.state.node_id} | Layer {layer}")
                retrieve_action = RetrieveAction(
                    target_nodes=graph.get_neighbors(self.state.node_id),
                    info_type="all"
                )
                try:
                    with timeout(5):
                        retrieve_result = retrieve_action.execute(self, graph)
                        self._record_action_history(layer, retrieve_action, retrieve_result)
                        self._update_action_stats("success", "RetrieveAction")
                except TimeoutError:
                    print(f"⚠️ Fallback retrieve timed out")
                    self._update_action_stats("timeout")
                    return
                except Exception as e:
                    print(f"⚠️ Error in fallback retrieve: {e}")
                    self._update_action_stats("failure")
                    return

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

            fallback_prompt = self._format_fallback_label_prompt(self.state.text, self.state.memory)
            print(f"\n📦 [Fallback Prompt for Node {self.state.node_id}]:\n{fallback_prompt}")
            fallback_response = self.llm.generate_response(fallback_prompt)
            fallback_decision = self.llm.parse_action(fallback_response)

            # extract update from list if needed
            if isinstance(fallback_decision, list):
                for d in fallback_decision:
                    if isinstance(d, dict) and d.get("action_type") == "update":
                        fallback_decision = d
                        break
                else:
                    fallback_decision = {"action_type": "no_op"}

            print(f"🎯 [Fallback Result]: {fallback_decision}")

            if isinstance(fallback_decision, dict) and fallback_decision.get("action_type") == "update":
                fallback_action = self._create_action(fallback_decision, graph)
                if fallback_action:
                    try:
                        with timeout(5):
                            fallback_result = fallback_action.execute(self, graph)
                            self._record_action_history(layer, fallback_action, fallback_result)
                            self._update_action_stats("success", "UpdateAction")
                    except TimeoutError:
                        print(f"⚠️ Fallback update timed out")
                        self._update_action_stats("timeout")
                    except Exception as e:
                        print(f"⚠️ Error in fallback update: {e}")
                        self._update_action_stats("failure")

    def _record_action_history(self, layer: int, action: Action, result: Any):
        """记录动作执行历史"""
        self.action_history.append({
            "layer": layer,
            "action": action.__class__.__name__,
            "result": result,
            "timestamp": time.time()
        })

    def _update_action_stats(self, status: str, action_type: Optional[str] = None):
        """更新动作执行统计信息"""
        self.action_stats[status] += 1
        if action_type:
            self.action_stats["by_type"][action_type] = \
                self.action_stats["by_type"].get(action_type, 0) + 1

    def _create_action(self, decision: Dict[str, Any], graph: 'AgenticGraph') -> Optional[Action]:
        if isinstance(decision, dict):
            action_type = decision.get("action_type", "no_op")
        elif isinstance(decision, Action):
            return decision
        elif isinstance(decision, (int, str)) and str(decision).isdigit():
            # 处理单个节点 ID 的情况
            return RetrieveAction([int(decision)], "all")
        elif isinstance(decision, list) and all(str(x).isdigit() for x in decision):
            # 处理节点 ID 列表的情况
            return RetrieveAction([int(x) for x in decision], "all")
        else:
            return NoOpAction()

        if getattr(self, "skip_update", False) and action_type == "update":
            return NoOpAction()

        if action_type == "retrieve":
            target_nodes = decision.get("target_nodes", [])
            # 确保 target_nodes 是整数列表
            if isinstance(target_nodes, (int, str)) and str(target_nodes).isdigit():
                target_nodes = [int(target_nodes)]
            elif isinstance(target_nodes, list):
                target_nodes = [int(x) for x in target_nodes if str(x).isdigit()]
            return RetrieveAction(target_nodes, decision.get("info_type", "text"))
        elif action_type == "rag_query":
            query = decision.get("query")
            if isinstance(query, str) and query.isdigit():
                query = int(query)
            if not isinstance(query, int):
                print(f"⚠️ Invalid rag_query input: {query}")
                return NoOpAction()
            return RAGAction(query, decision.get("top_k", 5))
        elif action_type == "broadcast":
            target_nodes = decision.get("target_nodes", [])
            # 确保 target_nodes 是整数列表
            if isinstance(target_nodes, (int, str)) and str(target_nodes).isdigit():
                target_nodes = [int(target_nodes)]
            elif isinstance(target_nodes, list):
                target_nodes = [int(x) for x in target_nodes if str(x).isdigit()]
            return BroadcastAction(target_nodes, torch.tensor([len(str(decision.get("message", "")))], dtype=torch.float))
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
        prompt = "You are a label prediction agent for research paper classification.\n\n"
        prompt += f"Text to classify:\n\"{node_text}\"\n\n"
        
        # 添加标签定义，使用统一的格式
        prompt += "Label Definitions (Use EXACTLY these labels in your response):\n"
        prompt += "Label_0 (Theory) - Fundamental research, mathematical proofs, theoretical frameworks\n"
        prompt += "Label_1 (Neural_Networks) - Deep learning, neural network architectures, backpropagation\n"
        prompt += "Label_2 (Probabilistic_Methods) - Statistical models, probability theory, Bayesian methods\n"
        prompt += "Label_3 (Case_Based) - Example-based reasoning, case studies, analogical reasoning\n"
        prompt += "Label_4 (Genetic_Algorithms) - Evolutionary computation, genetic programming, optimization\n"
        prompt += "Label_5 (Reinforcement_Learning) - Learning from rewards, policy optimization, MDPs\n"
        prompt += "Label_6 (Rule_Learning) - Symbolic rules, decision trees, rule extraction\n\n"
        
        if memory:
            prompt += "Memory items (labeled examples):\n"
            for m in memory:
                label = m.get("label", None)
                text = m.get("text", "")
                if label is not None:
                    label_str = inv_label_vocab.get(label, f"Label_{label}")
                    short_text = text[:60] + "..." if len(text) > 60 else text
                    prompt += f"- \"{short_text}\" — label: {label_str}\n"
        else:
            prompt += "Memory items: (No memory available)\n"

        prompt += "\nAnalysis Steps:\n"
        prompt += "1. Identify key terms in the text:\n"
        prompt += "   - Look for method names (e.g., HMM, neural networks, genetic algorithms)\n"
        prompt += "   - Look for mathematical concepts (e.g., probability, optimization, convex)\n"
        prompt += "   - Look for research areas (e.g., bioinformatics, machine learning)\n\n"
        
        prompt += "2. Match with label definitions:\n"
        prompt += "   - If text mentions specific methods, match with corresponding label\n"
        prompt += "   - If text focuses on mathematical foundations, consider Label_0 (Theory)\n"
        prompt += "   - If text discusses optimization, consider Label_2 (Probabilistic_Methods) or Label_4 (Genetic_Algorithms)\n"
        prompt += "   - If text is about learning from examples, consider Label_3 (Case_Based)\n\n"
        
        prompt += "3. Decision Rules:\n"
        prompt += "   - If text matches memory examples, use the same label\n"
        prompt += "   - If text mentions probability/statistics, prefer Label_2 (Probabilistic_Methods)\n"
        prompt += "   - If text is about optimization, check if it's probabilistic or evolutionary\n"
        prompt += "   - If uncertain, choose the label that best matches the primary methodology\n\n"
        
        prompt += "4. Common Patterns:\n"
        prompt += "   - Protein sequence analysis often uses Label_2 (Probabilistic_Methods)\n"
        prompt += "   - Convex optimization often relates to Label_2 (Probabilistic_Methods)\n"
        prompt += "   - Heuristics in ML often relate to the underlying method type\n\n"
        
        prompt += "IMPORTANT: Always use the EXACT label format (e.g., 'Label_2' not 'Probabilistic_Methods')\n"
        prompt += "Respond strictly in JSON format:\n{\"action_type\": \"update\", \"predicted_label\": \"label_string\"}\n"
        label_list = ", ".join(f'\"{v}\"' for v in inv_label_vocab.values())
        prompt += f"Allowed labels: [{label_list}]\n"
        
        return prompt

    def _format_simple_fallback_action_prompt(self, context: Dict[str, Any]) -> str:
        return f"Based on context {context}, choose one action: retrieve, broadcast, update, or rag_query."
