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
        node_id = context["node_id"]
        layer = context["layer"]
        text = context.get("text", "")
        neighbors = context["neighbors"]
        total_neighbors = context["total_neighbors"]
        messages = context.get("messages", [])
        retrieved_data = context.get("retrieved_data", {})
        memory = context.get("memory", [])

        seen_nodes = set(retrieved_data.keys())
        for m in memory:
            if isinstance(m, dict):
                result = m.get("result", {})
                target_nodes = result.get("target_nodes", [])
                if isinstance(target_nodes, int):
                    seen_nodes.add(target_nodes)
                elif isinstance(target_nodes, list):
                    seen_nodes.update(target_nodes)

        available_nodes = sorted(set(neighbors) - seen_nodes)

        prompt = f"""
You are an intelligent agent for Node {node_id} in a graph.

Your goal is to decide the best next action based on the current layer and known information.

## Node Info:
- Layer: {layer}
- Text: \"{text}\"
- Neighbors: {neighbors}
- Available nodes to retrieve from (excluding already seen): {available_nodes}
"""

        if retrieved_data:
            prompt += "\n## Retrieved Data:\n"
            for nid, val in list(retrieved_data.items())[:3]:
                prompt += f"- Node {nid}: {val}\n"
            if len(retrieved_data) > 3:
                prompt += f"(and {len(retrieved_data) - 3} more)\n"

        if messages:
            prompt += "\n## Messages Received:\n"
            for msg in messages:
                preview = msg.get("content_preview", "[no preview]")
                prompt += f"- From Node {msg['from']} (Layer {msg['layer']}): {preview}\n"

        if memory:
            prompt += "\n## Memory:\n"
            for i, content in enumerate(memory[:3]):
                prompt += f"- Memory #{i} (Layer {content.get('layer', '?')}):"
                if "label" in content:
                    prompt += f" label={content['label']}"
                if "action" in content:
                    prompt += f" action={content['action']}"
                if "result" in content and isinstance(content["result"], dict):
                    preview = str(content["result"])[:60].replace("\n", " ")
                    prompt += f" result={preview}"
                prompt += "\n"

        prompt += """

## Question:
Can you now classify this node? If yes, choose `update`. If more info is needed, choose `retrieve`. If unsure, choose `no_op`.

## Label Index Mapping:
0: Case_Based
1: Genetic_Algorithms
2: Neural_Networks
3: Probabilistic_Methods
4: Reinforcement_Learning
5: Rule_Learning
6: Theory

## Action Options (respond with JSON only):
1. Retrieve → {"action_type": "retrieve", "target_nodes": [IDs], "info_type": "text"}
2. Broadcast → {"action_type": "broadcast", "target_nodes": [IDs], "message": "..."}
3. Update label → {"action_type": "update", "predicted_label": label_id}
4. No-op → {"action_type": "no_op"}
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
