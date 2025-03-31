"""
Refactored LLM interface that selects backend (mock or remote) based on config.
"""

import json
import random
import re
import requests
from typing import Dict, Any, Optional
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
        return self._parse_action(self.generate_response(self._format_action_prompt(context)))

    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        response = self.generate_response(self._format_layer_prompt(context))
        return "continue" in response.lower()

    def _format_action_prompt(self, context: Dict[str, Any]) -> str:
        node_id = context["node_id"]
        layer = context["layer"]
        features = context["features"]
        neighbors = context["neighbors"][:10] if len(context["neighbors"]) > 10 else context["neighbors"]
        total_neighbors = context["total_neighbors"]
        messages = context.get("messages", [])
        retrieved_data = context.get("retrieved_data", {})
        memory = context.get("memory", {})

        prompt = f"""You are an intelligent node agent (Node {node_id}) in a graph neural network.
    Based on your current state and observations, you need to decide what action to take.

    Current Layer: {layer}
    Your Features: {features}
    Your Neighbors: {neighbors}{' (truncated)' if total_neighbors > len(neighbors) else ''}
    Total Neighbors: {total_neighbors}

    """

        if retrieved_data:
            prompt += "Retrieved data from previous actions:\n"
            preview_count = min(3, len(retrieved_data))
            previewed_items = list(retrieved_data.items())[:preview_count]
            for neighbor_id, data in previewed_items:
                prompt += f"- Node {neighbor_id}: {data}\n"
            if len(retrieved_data) > preview_count:
                prompt += f"(and {len(retrieved_data) - preview_count} more nodes)\n"
            prompt += "\n"

        if messages:
            prompt += "Recent messages:\n"
            for msg in messages:
                preview = msg.get("content_preview", "[no preview]")
                prompt += f"- From Node {msg['from']} (Layer {msg['layer']}): content preview {preview}\n"
            prompt += "\n"

        if memory:
            prompt += "Memory from previous layers:\n"
            for node_id, content in list(memory.items())[:3]:  # limit preview
                prompt += f"- Node {node_id} (Layer {content['source_layer']}): "
                if "features" in content:
                    prompt += f"features: {content['features'][:5]}... "
                if "label" in content:
                    prompt += f"label: {content['label']} "
                prompt += "\n"
            prompt += "\n"

        prompt += """            
    Available actions:
    1. retrieve - Retrieve information from any nodes
    Example: {"action_type": "retrieve", "target_nodes": [some_node_ids], "info_type": "features"}
    - info_type can be "features", "label", or "both"
    - You may retrieve information from any node in the graph (not just your neighbors).
    -You are more likely to have relevant knowledge about your neighbors, but you're free to query any node.


    2. broadcast - Send a message to neighbors
    Example: {"action_type": "broadcast", "target_nodes": [some_node_ids], "message": [0.5, 0.3, 0.7]}

    3. update - Update your own state
    Example: {"action_type": "update", "predicted_label": some_label}
    - When deciding to `update`, make sure you have enough evidence to predict your label.
    - You should compare your own features with those from other nodes (retrieved or in memory). Try to find the most similar ones based on vector similarity (e.g., cosine similarity or Euclidean distance).
    - Choose the label from the most similar node(s) if they appear trustworthy or consistent.
    - You also have access to your internal memory. Memory stores information from earlier steps, such as:
        - features retrieved from other nodes
        - labels or messages you've seen before
    - Use memory to accumulate evidence over time. For example, if you retrieved 3 neighbors' features in Layer 0 and 2 more in Layer 1, you should consider all 5 before updating.
    - You can rely on both memory and newly retrieved data to make a confident prediction.
    - If the evidence is strong (e.g., several similar nodes with the same label), you can safely update. Otherwise, retrieve more or choose `no_op` to wait.
    * Summary:
    - Use your features + memory + retrieved features.
    - Compare vectors.
    - Pick the most likely label.
    - If unsure, wait or retrieve again, don't guess and update randomly.


    4. no_op - Do nothing
    Example: {"action_type": "no_op"}
    If you think you have enough information, you can choose to do nothing.

    Choose the action that best fits your current situation and goals. Respond with a valid JSON object.
    """
        return prompt


    def _format_layer_prompt(self, context: Dict[str, Any]) -> str:
        return f"""You are the controller for a graph neural network.\n\nThe network has completed layer {context['current_layer']} of processing.\n\nMax layers: {context['max_layers']}\nCurrent layer: {context['current_layer']}\n\nBased on the progress, decide whether to:\n1. Continue to the next layer\n2. End processing and output final results\n\nRespond with either \"continue\" or \"end\".\n"""

    def _parse_action(self, response: str) -> Dict[str, Any]:
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                return json.loads(response[start_idx:end_idx+1])
        except Exception as e:
            print(f"[RemoteLLMInterface] Failed to parse response: {e}")
        return {"action_type": "no_op"}


class MockLLMInterface(BaseLLMInterface):
    def generate_response(self, prompt: str) -> str:
        return "continue" if "layer" in prompt.lower() else json.dumps({"action_type": "no_op"})

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("layer", 0) == 0 and context.get("neighbors"):
            sample = random.sample(context["neighbors"], min(3, len(context["neighbors"])))
            return {"action_type": "retrieve", "target_nodes": sample, "info_type": "features"}
        return {"action_type": "update", "predicted_label": random.randint(0, 39)}

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
