import json
import random
import re
import requests
from typing import Dict, Any, Optional, List
import config


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
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.95
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            print("🔁 Raw response from vLLM:", response.text)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[RemoteLLMInterface] Request failed: {e}")
            return "no_op"

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        response = self.generate_response(self._format_action_prompt(context))
        parsed = self._parse_action(response)
        if parsed.get("action_type") == "retrieve":
            parsed["target_nodes"] = [
                int(re.sub(r"[^\d]", "", str(nid))) for nid in parsed.get("target_nodes", [])
            ]
        return parsed

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        response = self.generate_response(self._format_layer_prompt(context))
        return "continue" in response.lower()

    def _format_action_prompt(self, context: Dict[str, Any]) -> str:
        from data.cora.label_vocab import inv_label_vocab

        node_id = context["node_id"]
        layer = context["layer"]
        text = context.get("text", "")
        neighbors = context["neighbors"]
        total_neighbors = context["total_neighbors"]
        messages = context.get("messages", [])
        retrieved_data = context.get("retrieved_data", {})
        memory = context.get("memory", [])

        # NEW: label & broadcast status
        node_label = context.get("label") or context.get("predicted_label")
        has_broadcasted = context.get("has_broadcasted", False)

        # Build seen & available nodes
        seen_nodes = set(retrieved_data.keys())
        for m in memory:
            if isinstance(m, dict):
                result = m.get("result", {})
                target_nodes = result.get("target_nodes", [])
                if isinstance(target_nodes, int):
                    seen_nodes.add(target_nodes)
                elif isinstance(target_nodes, list):
                    seen_nodes.update(target_nodes)
        flat_seen_nodes = list(sorted(seen_nodes))
        available_nodes = sorted(set(neighbors) - seen_nodes)

        # Instruction
        prompt = f"""
    You are an intelligent node agent responsible for predicting the correct label for a node in a scientific graph.

    ## Your State:
    - Node ID: {node_id}
    - Layer: {layer}
    - Your Text:
    \"{text}\"
    - Neighbors: {neighbors if neighbors else 'None'}
    - Available nodes to retrieve (excluding seen): {available_nodes if available_nodes else 'None'}
    """

        # Label prediction section
        label_list = ", ".join([f"{i}. {label}" for i, label in inv_label_vocab.items()])
        prompt += f"""

    ## Label Categories:
    You must classify the node into one of the following categories:
    {label_list}
    """

        # Memory examples with label
        labeled_examples = [m for m in memory if m.get("label") is not None and m.get("text")]
        if labeled_examples:
            prompt += "\n## Memory Examples with Known Labels:\n"
            prompt += "Refer to the following labeled nodes to help predict the label of the current node:\n"
            for i, ex in enumerate(labeled_examples[:5]):
                lbl = inv_label_vocab.get(ex["label"], "?")
                prompt += f"{i+1}. [{lbl}] \"{ex['text'][:60]}\"\n"

        # Received messages summary
        if messages:
            prompt += "\n## Messages Received:\n"
            for msg in messages:
                preview = msg.get("content_preview", "[no preview]")
                prompt += f"- From Node {msg['from']} (Layer {msg['layer']}): Preview={preview}\n"

        if retrieved_data:
            prompt += "\n## Retrieved Data (from previous steps):\n"
            for nid, val in list(retrieved_data.items())[:3]:
                prompt += f"- Node {nid}: {val}\n"
            if len(retrieved_data) > 3:
                prompt += f"(and {len(retrieved_data) - 3} more)\n"

        # ✅ NEW SECTION: encourage broadcasting if node has label and hasn't broadcasted yet
        if node_label is not None and not has_broadcasted:
            prompt += f"""

    ⚠️ You already have a label: "{node_label}". You may consider broadcasting this label and your text to your neighbors to help them in their predictions.
    """

        # Final instruction
        prompt += """

    ## Decide Your Next Action
    Based on your text and memory, you should select one of the following actions:

    1. "retrieve": get information from other nodes
    - Format: {"action_type": "retrieve", "target_nodes": [IDs], "info_type": "text"}

    2. "broadcast": send a message to neighbors
    - Format: {"action_type": "broadcast", "target_nodes": [IDs], "message": "some message"}

    3. "update": decide your label
    - Format: {"action_type": "update", "predicted_label": "label_string"}
    - ⚠️ Only use memory to infer your label. You **must** base the prediction only on nodes in memory with known labels.

    4. "no_op": take no action
    - Format: {"action_type": "no_op"}
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
        """
        Format a prompt for fallback label prediction using labeled examples from memory.
        
        Args:
            node_text: The text to classify
            memory: List of memory entries containing labeled examples
            
        Returns:
            Formatted prompt string
        """
        from data.cora.label_vocab import inv_label_vocab
        
        prompt = "You are given a scientific paper text and several labeled examples. Predict the most likely label.\n\n"
        prompt += f"Text to classify:\n\"{node_text}\"\n\n"
        prompt += "Labeled examples:\n"
        
        # Filter memory entries that have valid labels
        labeled_examples = [m for m in memory if m.get("label") is not None]
        
        for m in labeled_examples[:5]:  # Use at most 5 examples
            lbl = inv_label_vocab.get(m["label"], "?")
            txt = m["text"][:60] + "..." if len(m["text"]) > 60 else m["text"]
            prompt += f"- [{lbl}] \"{txt}\"\n"
            
        prompt += "\nRespond with:\n{\"action_type\": \"update\", \"predicted_label\": \"label_string\"}\n"
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


class LLMInterface(BaseLLMInterface):
    def __init__(self, model_name: str = config.LLM_MODEL):
        self.backend = config.LLM_BACKEND
        if self.backend == "mock":
            print("[LLMInterface] Using Mock LLM backend.")
            self.impl = MockLLMInterface()
        elif self.backend == "remote":
            print(f"[LLMInterface] Using Remote LLM backend: {config.REMOTE_LLM_ENDPOINT}")
            self.impl = RemoteLLMInterface(config.REMOTE_LLM_ENDPOINT, model_name)
        else:
            raise ValueError(f"Unsupported LLM_BACKEND: {self.backend}")

    def generate_response(self, prompt: str) -> str:
        print("📤 [DEBUG] Prompt being sent to vLLM:\n", prompt)
        return self.impl.generate_response(prompt)

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl.decide_action(context)

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        return self.impl.determine_next_layer(context)

    def _format_fallback_label_prompt(self, node_text: str, memory: List[Dict[str, Any]]) -> str:
        return self.impl._format_fallback_label_prompt(node_text, memory)

    def parse_action(self, response: str) -> Dict[str, Any]:
        return self.impl._parse_action(response)
