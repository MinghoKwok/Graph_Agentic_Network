import json
import random
import re
import requests
from typing import Dict, Any, Optional, List
import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, Any, Optional, List, Union
import datetime

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
        self.debug_logs_dir = os.path.join(config.RESULTS_DIR, "debug_logs")
        os.makedirs(self.debug_logs_dir, exist_ok=True)

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
            print("🔍 Full LLM raw output:")
            print(result["choices"][0]["message"]["content"])
            
            # 记录响应到日志
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(self.debug_logs_dir, "llm_responses.txt"), "a", encoding="utf-8") as log_file:
                log_file.write(f"\n\n🔁 [LLM Response] | {timestamp}:\n")
                log_file.write(json.dumps(result, indent=2, ensure_ascii=False))
                log_file.write("\n" + "=" * 80)
            
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
        - You MUST base your decision only on the definitions of the labels and the memory nodes with known labels.
        - You should ALWAYS follow this action with a "broadcast" to share your label with neighbors.
"""

        # 注入标签定义
        label_definition = """
Here are the definitions of the labels, which are helpful for you to predict your label:

[label=Theory]
- Focuses on foundational models and algorithms underlying machine learning.
- Explores formal frameworks like PAC learning, VC dimension, and computational complexity.
- Addresses theoretical limitations and generalization guarantees.
- Includes studies on learnability and approximation strategies.
- Emphasizes conceptual clarity and rigorous analysis.

[label=Neural_Networks]
- Investigates layered architectures for pattern recognition and learning.
- Covers models like CNNs, RNNs, and feedforward networks.
- Learns via backpropagation and activation tuning.
- Applied in tasks such as vision, sequence modeling, and signal processing.
- Inspired by biological systems and deep representations.

[label=Case_Based]
- Solves problems by referencing past similar examples.
- Stores and retrieves previous cases for reasoning.
- Adapts old solutions to new problems.
- Applies to diagnosis, design support, and decision-making.
- Emphasizes example-driven and explainable inference.

[label=Genetic_Algorithms]
- Uses evolution-inspired methods to optimize solutions.
- Operates with selection, crossover, and mutation.
- Evolves rule sets, classifiers, or architectures over time.
- Excels in complex search spaces with rugged landscapes.
- Highlights robustness and adaptive search behavior.

[label=Probabilistic_Methods]
- Models uncertainty through probability and Bayesian reasoning.
- Includes graphical models, sampling, and inference.
- Handles noisy or incomplete data in decision-making.
- Applied in diagnosis, prediction, and structured reasoning.
- Combines interpretability with statistical rigor.

[label=Reinforcement_Learning]
- Learns from interactions with environment via rewards.
- Balances exploration and exploitation to find optimal policies.
- Formalized as MDPs with agents and states.
- Used in robotics, control, and game-playing systems.
- Focuses on long-term decision-making under uncertainty.

[label=Rule_Learning]
- Extracts symbolic rules like "if-then" from training data.
- Produces interpretable and compact decision logic.
- Applies logical reasoning for classification tasks.
- Suitable for expert systems and knowledge discovery.
- Optimizes rule accuracy, generality, and simplicity.

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
    1. If you have a predicted label, you can choose to broadcast it or continue to retrieve nodes with labels.
    2. If you don't have a predicted label, think like a planner: first gather evidence (retrieve, rag_query), then make a decision (update), and finally help others (broadcast).
    Think about the following:
    - If you cannot predict your label yet, need more context to predict your label → `retrieve`, `rag_query`
    - Are you confident to predict your label? → `update`
    - Have you shared your label or predicted label with neighbors? → `broadcast`
    - Only broadcast if you have a predicted label or training label, AND your memory is not empty. If not, choose "retrieve" or "rag_query" first.
    - If any neighbors already have predicted labels, it is recommended to retrieve from them first.
    """

        print("📤 [DEBUG] Prompt being sent to LLM:\n", prompt)
        
        # 记录 prompt 到日志
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.debug_logs_dir, f"node_{node_id}")
        os.makedirs(log_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, f"layer_{layer}_{timestamp}.txt"), "w", encoding="utf-8") as log_file:
            log_file.write(f"📤 [DEBUG] Prompt for Node {node_id} | Layer {layer} | {timestamp}:\n")
            log_file.write(prompt)
            log_file.write("\n" + "=" * 80)
        
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
        return self.impl.generate_response(prompt)

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl.decide_action(context)

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        return self.impl.determine_next_layer(context)

    def parse_action(self, response: str) -> Dict[str, Any]:
        return self.impl._parse_action(response)

    def _format_action_prompt(self, context: Dict[str, Any], graph: 'AgenticGraph' = None) -> str:
        return self.impl._format_action_prompt(context)