import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import importlib

from gan.actions import Action, RetrieveAction, RAGAction, BroadcastAction, UpdateAction, NoOpAction
from config import DEBUG_STEP_SUMMARY, DEBUG_MESSAGE_TRACE, NUM_LAYERS, DEBUG_FORCE_FALLBACK, DATASET_NAME
from gan.utils import has_memory_entry
import config

def get_label_vocab(dataset_name: str):
    """动态导入指定数据集的 label_vocab"""
    module = importlib.import_module(f"data.{dataset_name}.label_vocab")
    return module.label_vocab, module.inv_label_vocab

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
    def __init__(self, state: NodeState, llm_interface: 'LLMInterface', dataset_name: str = DATASET_NAME):
        self.state = state
        self.llm = llm_interface
        self.retrieved_data = {}
        self.memory = {}
        self.dataset_name = dataset_name
        self.label_vocab, self.inv_label_vocab = get_label_vocab(dataset_name)

    def step(self, graph: 'AgenticGraph', layer: int):
        neighbors = graph.get_neighbors(self.state.node_id)
        labeled_neighbors = [nid for nid in neighbors if graph.get_node(nid).state.label is not None]
        # import pdb; pdb.set_trace()
        context = {
            "node_id": self.state.node_id,
            "text": self.state.text,
            "label": self.state.label.item() if self.state.label is not None else None,
            "layer": layer,
            "memory": self.state.memory,
            "neighbors": neighbors,
            "total_neighbors": len(neighbors),
            "labeled_neighbors": labeled_neighbors
        }

        self.skip_update = hasattr(graph, "train_idx") and self.state.node_id in graph.train_idx
        action_prompt = self.llm._format_action_prompt(context)

        try:
            response = self.llm.generate_response(action_prompt)
            action = self.llm.parse_action(response) or self.llm.parse_action(self.llm.generate_response(self._format_simple_fallback_action_prompt(context)))
            if isinstance(action, dict):
                action = self._create_action(action, graph)
            if action:
                result = action.execute(self, graph)
                if isinstance(action, RAGAction) and isinstance(result, dict) and "results" in result:
                    print(f"\n🔍 Processing RAG results for node {self.state.node_id}:")
                    print(f"Results: {result['results']}")
                    for node_id, node_info in result["results"].items():
                        print(f"\nProcessing node {node_id}:")
                        print(f"Node info: {node_info}")
                        if isinstance(node_info, dict):
                            memory_entry = {
                                "layer": layer,
                                "action": "RAG",
                                "text": str(node_info.get("text", "")),
                                "label": int(node_info.get("label", -1)) if node_info.get("label") is not None else None,
                                "label_text": self.inv_label_vocab.get(int(node_info.get("label", -1)), str(node_info.get("label", -1))) if node_info.get("label") is not None else None,
                                "source": int(node_id),
                                "source_type": "rag"
                            }
                            print(f"Memory entry: {memory_entry}")
                            if not has_memory_entry(self, memory_entry):
                                self.state.memory.append(memory_entry)
                                print(f"✅ Added to memory")
                            else:
                                print(f"⚠️ Entry already exists in memory")
                if isinstance(result, dict) and result.get("action_type") == "retrieve":
                    for node_id, node_info in result.get("results", {}).items():
                        if isinstance(node_info, dict) and node_info.get("text"):
                            memory_entry = {
                                "layer": layer,
                                "action": "Retrieve",
                                "text": str(node_info['text']),
                                "label": int(node_info.get('label')) if node_info.get('label') is not None else None,
                                "label_text": self.inv_label_vocab.get(int(node_info.get('label')), str(node_info.get('label'))) if node_info.get('label') is not None else None,
                                "source": int(node_id),
                                "source_type": "retrieved"
                            }
                            if not has_memory_entry(self, memory_entry):
                                self.state.memory.append(memory_entry)
        except Exception as e:
            print(f"⚠️ Error in agent step: {e}")

        # Fallback 强制 Retrieve + RAG
        ## 强制 Retrieve
        retrieve_action = RetrieveAction(target_nodes=labeled_neighbors, info_type="all")
        retrieve_result = retrieve_action.execute(self, graph)
        if isinstance(retrieve_result, dict) and "results" in retrieve_result:
            for node_id, node_info in retrieve_result["results"].items():
                if isinstance(node_info, dict) and node_info.get("text"):
                    memory_entry = {
                        "layer": layer,
                        "action": "Retrieve",
                        "text": str(node_info['text']),
                        "label": int(node_info.get('label')) if node_info.get('label') is not None else None,
                        "label_text": self.inv_label_vocab.get(int(node_info.get('label')), str(node_info.get('label'))) if node_info.get('label') is not None else None,
                        "source": int(node_id),
                        "source_type": "retrieved"
                    }
                    if not has_memory_entry(self, memory_entry):
                        self.state.memory.append(memory_entry)
        ## 强制 RAG
        rag_action = RAGAction(self.state.node_id, top_k=5)
        rag_result = rag_action.execute(self, graph)
        if isinstance(rag_result, dict) and "results" in rag_result:
            print(f"\n🔍 Processing fallback RAG results for node {self.state.node_id}:")
            print(f"Results: {rag_result['results']}")
            for node_id, node_info in rag_result["results"].items():
                print(f"\nProcessing node {node_id}:")
                print(f"Node info: {node_info}")
                if isinstance(node_info, dict):
                    memory_entry = {
                        "layer": layer,
                        "action": "RAG",
                        "text": str(node_info.get("text", "")),
                        "label": int(node_info.get("label", -1)) if node_info.get("label") is not None else None,
                        "label_text": self.inv_label_vocab.get(int(node_info.get("label", -1)), str(node_info.get("label", -1))) if node_info.get("label") is not None else None,
                        "source": int(node_id),
                        "source_type": "rag"
                    }
                    print(f"Memory entry: {memory_entry}")
                    if not has_memory_entry(self, memory_entry):
                        self.state.memory.append(memory_entry)
                        print(f"✅ Added to memory")
                    else:
                        print(f"⚠️ Entry already exists in memory")

        fallback_prompt = self._format_fallback_label_prompt(self.state.text, self.state.memory)
        print(f"\n📦 [Fallback Prompt for Node {self.state.node_id}]:\n{fallback_prompt}")
        fallback_response = self.llm.generate_response(fallback_prompt)
        fallback_decision = self.llm.parse_action(fallback_response)
        if isinstance(fallback_decision, dict) and fallback_decision.get("action_type") == "update":
            fallback_action = self._create_action(fallback_decision, graph)
            if fallback_action:
                fallback_action.execute(self, graph)

    def _create_action(self, decision: Dict[str, Any], graph: 'AgenticGraph') -> Optional[Action]:
        action_type = decision.get("action_type", "no_op") if isinstance(decision, dict) else "no_op"
        if getattr(self, "skip_update", False) and action_type == "update":
            return NoOpAction()
        if action_type == "retrieve":
            return RetrieveAction(decision.get("target_nodes", []), decision.get("info_type", "text"))
        elif action_type == "rag_query":
            return RAGAction(self.state.node_id, decision.get("top_k", 5))
        elif action_type == "broadcast":
            return BroadcastAction(decision.get("target_nodes", []), torch.tensor([len(str(decision.get("message", "")))], dtype=torch.float))
        elif action_type == "update":
            label_value = decision.get("predicted_label")
            label_id = self.label_vocab.get(label_value, -1) if isinstance(label_value, str) else label_value
            return UpdateAction({"predicted_label": torch.tensor(label_id)}) if label_id >= 0 else NoOpAction()
        return NoOpAction()

    def _format_fallback_label_prompt(self, node_text: str, memory: List[Dict[str, Any]], top_k: int = 5) -> str:
        from difflib import SequenceMatcher
        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()
        print(f"\n👀 Memory: {memory}")
        retrieved_memory = [m for m in memory if m.get("source_type") == "retrieved" and m.get("label") is not None and m.get("text")]
        rag_memory = [m for m in memory if m.get("source_type") == "rag" and m.get("label") is not None and m.get("text")]
        print(f"\n👀 Retrieved memory: {retrieved_memory}")
        print(f"\n👀 RAG memory: {rag_memory}")
        retrieved_top = sorted(retrieved_memory, key=lambda m: similarity(node_text.lower(), m["text"].lower()), reverse=True)[:5]
        if retrieved_top:
            rag_top = sorted(rag_memory, key=lambda m: similarity(node_text.lower(), m["text"].lower()), reverse=True)[:5]
        else:
            rag_top = sorted(rag_memory, key=lambda m: similarity(node_text.lower(), m["text"].lower()), reverse=True)[:5]
        prompt = "You are a label prediction agent.\n\n"
        prompt += f"Text to classify:\n\"{node_text.strip()}\"\n\n"
        memory_combined = retrieved_top + rag_top
        if memory_combined:
            prompt += "Memory items:\n"
            for m in memory_combined:
                label_str = self.inv_label_vocab.get(m["label"], f"Label_{m['label']}")
                short_text = m["text"].strip().replace("\n", " ")
                short_text = short_text[:100] + "..." if len(short_text) > 100 else short_text
                prompt += f'- "{short_text}" — label: {label_str}\n'
        else:
            prompt += "Memory items: (No memory available)\n"
        
        if config.DATASET_NAME == "citeseer" or config.DATASET_NAME == "Cora":
            prompt += "========================\n"
            prompt += "✅ Classification Rules (Follow Strictly):\n"
            prompt += "🔵 Rule 1: Majority Label Rule\n"
            prompt += "If one label appears more times than any other, you must assign that label.\n"
            prompt += "Even if another label looks slightly more semantically similar, DO NOT override.\n"

            prompt += "Example:\n"
            prompt += "Label_2 appears 5 times\n"
            prompt += "→ You must return Label_2\n"

            prompt += "🟡 Rule 2: Override Exception (Rare)\n"
            prompt += "You may override the majority ONLY IF ALL the following conditions are met:\n"
            prompt += "The current text is clearly incompatible with the majority label group;\n"
            prompt += "There is another label with at least 2 memory items;\n"
            prompt += "You provide this exact 3-part justification:\n"
            prompt += "State the majority label;\n"
            prompt += "Explain why the current text does not match it;\n"
            prompt += "Explain why the new label has better support from memory.\n"

            prompt += "⚠️ If you override without this format, your prediction is INVALID.\n"

            prompt += "🔴 Rule 3: No Overfitting / No Guessing\n"
            prompt += "Do NOT rely on vague words like “agent”, “system”, “architecture”.\n"

            prompt += "Match based on task, method, and topic.\n"

            prompt += "When unsure: prefer the majority label, or the lower-numbered label among ties.\n"

            prompt += "🧾 Final Output Format (JSON only):\n"
            prompt += "json\n"
            prompt += "Copy\n"
            prompt += "Edit\n"
            prompt += "{\n"
            prompt += "  \"action_type\": \"update\",\n"
            prompt += "  \"predicted_label\": \"Label_X\",\n"
            prompt += "  \"justification\": \"...\" \n"
            prompt += "}\n"
            prompt += "⚠️ DO NOT IGNORE THIS\n"
            prompt += "You are NOT doing freeform semantic matching.\n"
            prompt += "You are executing a rule-based classification protocol.\n"
            prompt += "Disregarding the majority label without structured override is a critical failure.\n"

            prompt += "✅ When the current text has some relevance to both groups, always prefer the majority label — even if it feels less “semantically elegant”."
        else:
            prompt += "\nPlease follow these steps in your analysis:\n"
            prompt += "1. Analyze the Current Node Text:\n   - Identify primary topics and application domain\n   - Determine the specific problem being solved\n   - Note core methodologies and algorithms\n"
            prompt += "2. Analyze Memory Examples:\n   - Understand application domains for each label\n   - Identify types of problems addressed\n   - Note underlying methodologies\n"
            prompt += "3. Compare and Weigh Evidence:\n   - Prioritize domain and problem alignment\n   - Evaluate methodological congruence\n   - Consider both domain-specific techniques and general paradigms\n   - Ensure holistic coherence in your decision\n"
            prompt += "4. Avoid over-reliance on isolated keywords\n\n"
            prompt += "Please think step by step: First analyze memory examples and their labels, then compare them to the input text. Identify the most semantically similar memory items and explain why. Finally, decide which label best matches and explain your reasoning."
        
        prompt += "\nRespond strictly in JSON format:\n{\"action_type\": \"update\", \"predicted_label\": \"label_string\"}\n"
        label_list = ", ".join(f'\"{v}\"' for v in self.inv_label_vocab.values())
        prompt += f"Allowed labels: [{label_list}]\n"
        return prompt

    def _format_simple_fallback_action_prompt(self, context: Dict[str, Any]) -> str:
        return f"Based on context {context}, choose one action: retrieve, broadcast, update, or rag_query."
