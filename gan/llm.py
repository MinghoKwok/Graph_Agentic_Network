import json
import random
import re
import requests
from typing import Dict, Any, Optional, List
import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, Any, Optional, List, Union

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gan.utils import get_labeled_examples, truncate_text
from data.cora.label_vocab import inv_label_vocab



class BaseLLMInterface:
    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        raise NotImplementedError


class RemoteLLMInterface(BaseLLMInterface):
    def __init__(self, endpoint: str, model_name: str):
        self.endpoint = endpoint
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        assert isinstance(prompt, str) and len(prompt.strip()) > 30, "Prompt seems too short or empty!"

        # 自动估算 prompt token 长度（粗略：单词数 × 1.3）
        approx_prompt_tokens = int(len(prompt.strip().split()) * 1.3)

        # 与你启动时设置的 max_model_len 保持一致
        max_model_len = config.MAX_MODEL_LEN
        max_tokens = max(min(max_model_len - approx_prompt_tokens, 512), 64)  # 最多 512，最少保留 64

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            print(f"🔁 Raw response from vLLM: {response.text[:200]}...")  # 前200字节预览，避免爆屏
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[RemoteLLMInterface] Request failed: {e}")
            return "no_op"

    def decide_action(self, context: Dict[str, Any], graph: 'AgenticGraph' = None) -> Dict[str, Any]:
        prompt = self._format_action_prompt(context, graph)
        response = self.generate_response(prompt)
        parsed = self._parse_action(response)
        if parsed.get("action_type") == "retrieve":
            parsed["target_nodes"] = [
                int(re.sub(r"[^\d]", "", str(nid))) for nid in parsed.get("target_nodes", [])
                if re.sub(r"[^\d]", "", str(nid)).isdigit()
            ]
        return parsed

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        response = self.generate_response(self._format_layer_prompt(context))
        return "continue" in response.lower()

    def _format_action_prompt(self, context: Dict[str, Any]) -> str:

        node_id = context["node_id"]
        layer = context["layer"]
        text = context.get("text", "")
        neighbors = context["neighbors"]
        total_neighbors = context["total_neighbors"]
        messages = context.get("messages", [])
        retrieved_data = context.get("retrieved_data", {})
        memory = context.get("memory", [])
        updated_neighbors = context.get("updated_neighbors", [])

        node_label = context.get("label") or context.get("predicted_label")
        has_broadcasted = context.get("has_broadcasted", False)

        # 只选择有标签的邻居
        labeled_neighbors = context.get("labeled_neighbors", [])
        
        prompt = ""

        prompt_intro = f"""
    You are Node {node_id} in a scientific citation network. Your task is to classify yourself into the correct research category based on your text and connections.
    """
        prompt += prompt_intro

        # 添加few-shot示例
        prompt += """
    ## Few-shot Examples of Label Prediction:

    Example 1:
    Memory:
    1. [Neural_Networks] "A novel deep learning approach for image classification..."
    2. [Reinforcement_Learning] "Q-learning based algorithm for game playing..."
    3. [Neural_Networks] "Convolutional neural networks for computer vision tasks..."

    Current Node Text:
    "Deep learning models for visual recognition tasks..."

    Prediction: Neural_Networks
    Reasoning: The current text focuses on deep learning and visual recognition, which closely matches the Neural_Networks examples in memory.

    Example 2:
    Memory:
    1. [Probabilistic_Methods] "Bayesian networks for uncertainty modeling..."
    2. [Neural_Networks] "Recurrent neural networks for sequence prediction..."
    3. [Probabilistic_Methods] "Markov models for time series analysis..."

    Current Node Text:
    "Hidden Markov models for speech recognition..."

    Prediction: Probabilistic_Methods
    Reasoning: The text discusses Markov models, which is a probabilistic method, matching the Probabilistic_Methods examples in memory.

    Example 3:
    Memory:
    1. [Neural_Networks] "Deep learning architectures for natural language processing..."
    2. [Theory] "Theoretical analysis of algorithm complexity..."
    3. [Neural_Networks] "Transformer models for sequence modeling..."

    Current Node Text:
    "Attention mechanisms in deep learning models for text understanding..."

    Prediction: Neural_Networks
    Reasoning: Although the text mentions theoretical concepts like attention mechanisms, the focus is on deep learning models and their application to text understanding, which closely matches the Neural_Networks examples in memory.
    """

        prompt_node_state = f"""
    ## Your State:
    - Node ID: {node_id}
    - Layer: {layer}
    - Your Text:
    \"{text}\"
    - Neighbors: {neighbors if neighbors else 'None'}
    - Available labeled neighbors to retrieve from: {labeled_neighbors if labeled_neighbors else 'None'}
    - Neighbors with predicted labels: {updated_neighbors if updated_neighbors else 'None'}
    """
        
        prompt += prompt_node_state

        label_list = ", ".join(f'"{v}"' for v in inv_label_vocab.values())
        update_action_block = f"""
        3. "update": decide your label *only* when the memory has enough information(labeled nodes, with text and label)
        - Format: {{"action_type": "update", "predicted_label": choose one of allowed labels: [{label_list}]}}
        - You MUST choose one of the allowed label strings exactly as listed.
        - You MUST base your decision only on memory nodes with known labels.
        - You should ALWAYS follow this action with a "broadcast" to share your label with neighbors.
"""


        # 使用精炼后的 labeled memory：text + label_text
        prompt_memory = ""
        memory_examples = get_labeled_examples(memory, top_k=5)
        if memory_examples:
            prompt_memory += "\n## Here are memory you have! Use such label-text pairs to predict your label:\n"
            for i, ex in enumerate(memory_examples):
                if isinstance(ex, str):
                    truncated = truncate_text(ex, max_words=50)
                    prompt_memory += f"{i+1}. {truncated}\n"

        prompt += prompt_memory
        if node_label is not None and not has_broadcasted:
            prompt += f"""⚠️ You already have a label: \"{node_label}\". You may consider broadcasting this label and your text to your neighbors to help them in their predictions."""

        prompt += """
    You are an autonomous agent with planning capabilities. You may perform multiple actions in sequence to achieve better results.

    ## Decide Your Next Action(s)
    Important: You are allowed and encouraged to return MULTIPLE actions in sequence. You MUST respond with a JSON array even if there's only one action. 
    Example of a valid response:
    ```json
    [
      {"action_type": "update", "predicted_label": "Neural_Networks"},
      {"action_type": "broadcast"}
    ]
    ```
    ```json
    [
      {"action_type": "retrieve", "target_nodes": [1, 2, 3], "info_type": "text"},
      {"action_type": "rag_query", "query": "machine learning", "top_k": 10}
    ]
    ```
    Invalid response:
    ```json
    {"action_type": "update", "predicted_label": "Neural_Networks"}
    ```

    ### Available Actions:

    1. "retrieve": get information from other nodes
    - Format: {"action_type": "retrieve", "target_nodes": [IDs], "info_type": "text"}

    2. "broadcast": send a message to neighbors if and *only* if you already have a label or predicted label
    - Format: {"action_type": "broadcast", "target_nodes": [IDs], "message": "some message"}
    - Use this *only* when you already have a label orpredicted label to share it with neighbors. 
    - You MUST NOT use "broadcast" unless you already have a label orpredicted label (i.e., after an "update" action).
    - So "update" action always works before "broadcast" in the same layer.

    """
        if context.get("label") is None:
            prompt += update_action_block

        prompt += """
    4. "rag_query": search globally for similar labeled nodes, can make up "retrieve" action
    - Format: {"action_type": "rag_query", "query": [Your node ID, e.g. 13/57], "top_k": number of nodes to retrieve}
    - Use this when you don't have enough informative neighbors or memory, and need global examples.
    - You must use your own node ID as the query.

    5. "no_op": take no action
    - Format: {"action_type": "no_op"}

    """

        prompt += """

    ## Planning Your Steps
    Think like a planner: first gather evidence (retrieve, rag_query), then make a decision (update), and finally help others (broadcast).
    Think about the following:
    - If you cannot predict your label yet, need more context to predict your label → `retrieve`, `rag_query`
    - Are you confident to predict your label? → `update`
    - Have you shared your label or predicted label with neighbors? → `broadcast`
    - Only broadcast if you have a predicted label or training label, AND your memory is not empty. If not, choose "retrieve" or "rag_query" first.
    - If any neighbors already have predicted labels, it is recommended to retrieve from them first.
    """

        return prompt

    def _format_layer_prompt(self, context: Dict[str, Any]) -> str:
        return f"""You are the controller for a graph neural network.\n\nThe network has completed layer {context['current_layer']} of processing.\n\nMax layers: {context['max_layers']}\nCurrent layer: {context['current_layer']}\n\nBased on the progress, decide whether to:\n1. Continue to the next layer\n2. End processing and output final results\n\nRespond with either \"continue\" or \"end\"."""

    def _parse_action(self, response: str) -> Dict[str, Any]:
        try:
            code_blocks = re.findall(r"```json\s*({.*?})\s*```", response, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r"({.*?})", response, re.DOTALL)
            for block in code_blocks:
                try:
                    cleaned = re.sub(r"//.*", "", block)
                    parsed = json.loads(cleaned)
                    if parsed.get("action_type") == "retrieve":
                        parsed["target_nodes"] = [
                            int(re.sub(r"[^\d]", "", str(nid)))
                            for nid in parsed.get("target_nodes", [])
                            if re.sub(r"[^\d]", "", str(nid)).isdigit()
                        ]
                    return parsed
                except Exception:
                    continue
        except Exception as e:
            print(f"[RemoteLLMInterface] Failed to parse response: {e}")
        return {"action_type": "no_op"}

    def _format_fallback_label_prompt(self, node_text: str, memory: List[Dict[str, Any]]) -> str:

        prompt = "You are given a scientific paper text and several labeled examples. Predict the most likely label.\n\n"
        prompt += f"Text to classify:\n\"{node_text}\"\n\n"
        prompt += "Labeled examples:\n"

        labeled_memory_examples = list({(m["label"], m["text"]): m for m in memory if m.get("label") is not None}.values())

        for m in labeled_memory_examples[:5]:
            lbl = inv_label_vocab.get(m["label"], "?")
            txt = m["text"][:60] + "..." if len(m["text"]) > 60 else m["text"]
            prompt += f"- [{lbl}] \"{txt}\"\n"

        prompt += "\nRespond with:\n{\"action_type\": \"update\", \"predicted_label\": \"label_string\"}\n"
        label_list = ", ".join(f'"{v}"' for v in inv_label_vocab.values())
        prompt += f"You must choose one of the allowed label strings exactly as listed: [{label_list}]"
        return prompt


class MockLLMInterface(BaseLLMInterface):
    def generate_response(self, prompt: str) -> str:
        return "continue" if "layer" in prompt.lower() else json.dumps({"action_type": "no_op"})

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("layer", 0) == 0 and context.get("neighbors"):
            sample = random.sample(context["neighbors"], min(3, len(context["neighbors"])))
            return {"action_type": "retrieve", "target_nodes": sample, "info_type": "text"}
        return {"action_type": "update", "predicted_label": random.randint(0, 6)}

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        return True

# 暂时不要了
# class FlanT5Interface(BaseLLMInterface):
#     def __init__(self, model_name: str):
#         print(f"[FlanT5Interface] Loading model: {model_name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=False)
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()
#             print("[FlanT5Interface] Model moved to GPU")
#         self.model.eval()

#     def generate_response(self, prompt: str) -> str:
#         inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         if torch.cuda.is_available():
#             inputs = {k: v.cuda() for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_length=512,
#                 num_beams=5,
#                 temperature=0.2,
#                 top_p=0.95,
#                 do_sample=True
#             )
        
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print("🔁 Raw response from Flan-T5:", response)
#         return response.strip()

#     def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
#         response = self.generate_response(self._format_action_prompt(context))
#         parsed = self._parse_action(response)
#         if parsed.get("action_type") == "retrieve":
#             parsed["target_nodes"] = [
#                 int(re.sub(r"[^\d]", "", str(nid))) for nid in parsed.get("target_nodes", [])
#             ]
#         return parsed

#     def determine_next_layer(self, context: Dict[str, Any]) -> bool:
#         response = self.generate_response(self._format_layer_prompt(context))
#         return "continue" in response.lower()

#     def _format_action_prompt(self, context: Dict[str, Any]) -> str:
#         from data.cora.label_vocab import inv_label_vocab

#         node_id = context["node_id"]
#         layer = context["layer"]
#         text = context.get("text", "")
#         neighbors = context["neighbors"]
#         # DEBUG: print neighbors' label and predicted_label
#         for nid in neighbors:
#             n = graph.get_node(nid)
#             print(f"[DEBUG] Neighbor {nid} | label: {n.state.label} | predicted_label: {n.state.predicted_label}")

#         total_neighbors = context["total_neighbors"]
#         messages = context.get("messages", [])
#         retrieved_data = context.get("retrieved_data", {})
#         memory = context.get("memory", [])
#         updated_neighbors = context.get("updated_neighbors", [])  # 获取已更新邻居列表

#         # NEW: label & broadcast status
#         node_label = context.get("label") or context.get("predicted_label")
#         has_broadcasted = context.get("has_broadcasted", False)

#         # Build seen & available nodes
#         seen_nodes = set(retrieved_data.keys())
#         for m in memory:
#             if isinstance(m, dict):
#                 result = m.get("result", {})
#                 target_nodes = result.get("target_nodes", [])
#                 if isinstance(target_nodes, int):
#                     seen_nodes.add(target_nodes)
#                 elif isinstance(target_nodes, list):
#                     seen_nodes.update(target_nodes)
#         flat_seen_nodes = list(sorted(seen_nodes))
#         available_nodes = sorted(set(neighbors) - seen_nodes)

#         # Instruction
#         prompt = f"""
# You are Node {node_id} in a scientific citation network. Your task is to classify yourself into the correct research category based on your text and connections.

#     ## Your State:
#     - Node ID: {node_id}
#     - Layer: {layer}
#     - Your Text:
#     \"{text}\"
#     - Neighbors: {neighbors if neighbors else 'None'}
#     - Available nodes to retrieve (excluding seen): {available_nodes if available_nodes else 'None'}
#     - Neighbors with predicted labels: {updated_neighbors if updated_neighbors else 'None'}
#     """

#         # Label prediction section
#         label_list = ", ".join([f"{i}. {label}" for i, label in inv_label_vocab.items()])
#         prompt += f"""

#     ## Label Categories:
#     You must classify the node into one of the following categories:
#     {label_list}
#     """

#         # Memory examples with label
#         labeled_examples = [m for m in memory if m.get("label") is not None and m.get("text")]
#         if labeled_examples:
#             prompt += "\n## Memory Examples with Known Labels:\n"
#             prompt += "Refer to the following labeled nodes to help predict the label of the current node:\n"
#             for i, ex in enumerate(labeled_examples[:5]):
#                 lbl = inv_label_vocab.get(ex["label"], "?")
#                 prompt += f"{i+1}. [{lbl}] \"{ex['text'][:60]}\"\n"

#         # Received messages summary
#         if messages:
#             prompt += "\n## Messages Received:\n"
#             for msg in messages:
#                 preview = msg.get("content_preview", "[no preview]")
#                 prompt += f"- From Node {msg['from']} (Layer {msg['layer']}): Preview={preview}\n"

#         if retrieved_data:
#             prompt += "\n## Retrieved Data (from previous steps):\n"
#             for nid, val in list(retrieved_data.items())[:3]:
#                 prompt += f"- Node {nid}: {val}\n"
#             if len(retrieved_data) > 3:
#                 prompt += f"(and {len(retrieved_data) - 3} more)\n"

#         # 显示从邻居收集到的其他节点信息
#         if retrieved_data:
#             collected_nodes = {}
#             for nid, val in retrieved_data.items():
#                 if "collected_nodes" in val:
#                     collected_nodes.update(val["collected_nodes"])
            
#             if collected_nodes:
#                 prompt += "\n## Collected Nodes from Neighbors' Memory:\n"
#                 prompt += "These are nodes information collected from your neighbors' memory (likely your 2nd or 3rd-degree connections):\n"
                
#                 for idx, (node_id, info) in enumerate(list(collected_nodes.items())[:5]):  # 限制显示的数量
#                     label_str = ""
#                     if "label" in info:
#                         label_val = info["label"]
#                         label_str = f"[Label: {inv_label_vocab.get(label_val, label_val)}]"
#                     elif "predicted_label" in info:
#                         label_val = info["predicted_label"]
#                         label_str = f"[Predicted: {inv_label_vocab.get(label_val, label_val)}]"
                    
#                     text_str = info.get("text", "")[:60]  # 限制文本长度
#                     prompt += f"- Node {node_id} {label_str}: \"{text_str}\"\n"
                
#                 if len(collected_nodes) > 5:
#                     prompt += f"(and {len(collected_nodes) - 5} more nodes)\n"

#         # ✅ NEW SECTION: encourage broadcasting if node has label and hasn't broadcasted yet
#         if node_label is not None and not has_broadcasted:
#             prompt += f"""

#     ⚠️ You already have a label: "{node_label}". You may consider broadcasting this label and your text to your neighbors to help them in their predictions.
#     """

#         # Final instruction
#         prompt += """

#     ## Decide Your Next Action(s)
#     Important: You are allowed and encouraged to return MULTIPLE actions in sequence. You MUST respond with a JSON array even if there's only one action. 
#     Example of a valid response:
#     ```json
#     [
#       {"action_type": "update", "predicted_label": "Neural_Networks"},
#       {"action_type": "broadcast"}
#     ]
#     ```
#     Invalid response:
#     ```json
#     {"action_type": "update", "predicted_label": "Neural_Networks"}
#     ```

#     ### Available Actions:

#     1. "retrieve": get information from other nodes
#     - Format: {"action_type": "retrieve", "target_nodes": [IDs], "info_type": "text"}

#     2. "broadcast": send a message to neighbors
#     - Format: {"action_type": "broadcast", "target_nodes": [IDs], "message": "some message"}

#     3. "update": decide your label when the memory has enough information(labeled nodes)
#     - Format: {"action_type": "update", "predicted_label": "label_string"}
#     - ⚠️ Only use memory to infer your label. You **must** base the prediction only on nodes in memory with known labels.

#     4. "rag_query": search globally for similar labeled nodes, can make up "retrieve" action
#     - Format: {"action_type": "rag_query", "query": "some query text", "top_k": 5}

#     5. "no_op": take no action
#     - Format: {"action_type": "no_op"}
    

#     ## Planning Your Steps
#     Think about the following:
#     - Do you need more context to predict your label? → `retrieve`, `rag_query`
#     - Are you confident to predict your label? → `update`
#     - Have you shared your label or predicted label with neighbors? → `broadcast`

#     """

#         return prompt


#     def _format_layer_prompt(self, context: Dict[str, Any]) -> str:
#         # 复用 RemoteLLMInterface 的 _format_layer_prompt 逻辑
#         return RemoteLLMInterface._format_layer_prompt(self, context)

#     def _parse_action(self, response: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
#         try:
#             code_blocks = re.findall(r"```json\\s*([\\s\\S]*?)\\s*```", response, re.DOTALL)
#             if not code_blocks:
#                 code_blocks = re.findall(r"\\[?\\{[\\s\\S]*?\\}?\\]?", response, re.DOTALL)

#             for block in code_blocks:
#                 try:
#                     cleaned = re.sub(r"//.*", "", block.strip())
#                     parsed = json.loads(cleaned)
#                     if isinstance(parsed, dict):
#                         return [parsed]
#                     elif isinstance(parsed, list):
#                         return parsed
#                 except Exception:
#                     continue
#         except Exception as e:
#             print(f"[RemoteLLMInterface] Failed to parse response: {e}")
#         return [{"action_type": "no_op"}]

    
#     def _format_fallback_label_prompt(self, node_text: str, memory: List[Dict[str, Any]]) -> str:
#         # 复用 RemoteLLMInterface 的 _format_fallback_label_prompt 逻辑
#         return RemoteLLMInterface._format_fallback_label_prompt(self, node_text, memory)


class LLMInterface(BaseLLMInterface):
    def __init__(self, model_name: str = config.LLM_MODEL):
        self.backend = config.LLM_BACKEND
        if model_name.startswith("google/flan-t5") or self.backend == "flan_local":  # ✅ 修改此处
            print(f"[LLMInterface] Using Flan-T5 LLM backend: {model_name}")
            self.impl = FlanT5Interface(model_name)
        elif self.backend == "mock":
            print("[LLMInterface] Using Mock LLM backend.")
            self.impl = MockLLMInterface()
        elif self.backend == "remote":
            print(f"[LLMInterface] Using Remote LLM backend: {config.REMOTE_LLM_ENDPOINT}")
            self.impl = RemoteLLMInterface(config.REMOTE_LLM_ENDPOINT, model_name)
        else:
            raise ValueError(f"Unsupported LLM_BACKEND: {self.backend}")

    def generate_response(self, prompt: str) -> str:
        print("📤 [DEBUG] Prompt being sent to LLM:\n", prompt)
        return self.impl.generate_response(prompt)

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl.decide_action(context)

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        return self.impl.determine_next_layer(context)

    def _format_fallback_label_prompt(self, node_text: str, memory: List[Dict[str, Any]]) -> str:
        return self.impl._format_fallback_label_prompt(node_text, memory)

    def parse_action(self, response: str) -> Dict[str, Any]:
        return self.impl._parse_action(response)

    def _format_action_prompt(self, context: Dict[str, Any], graph: 'AgenticGraph' = None) -> str:
        return self.impl._format_action_prompt(context)