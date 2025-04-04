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
        print(f"🔍 Neighbors: {neighbors}")

        # Build seen node set
        seen_nodes = set(retrieved_data.keys())
        for m in memory:
            if isinstance(m, dict):
                result = m.get("result", {})
                targets = result.get("target_nodes", [])
                if isinstance(targets, int):
                    seen_nodes.add(targets)
                elif isinstance(targets, list):
                    seen_nodes.update(targets)

        # Compute available candidates for retrieve
        available_nodes = sorted(set(neighbors) - seen_nodes)

        prompt = f"""
    You are Node {node_id} at Layer {layer}. Decide your next action.

    ## Your Text
    "{text}"

    ## Neighbors ({len(neighbors)} total)
    {neighbors[:10]}{' (truncated)' if len(neighbors) > 10 else ''}

    ## Seen Nodes ({len(seen_nodes)})
    {list(seen_nodes)[:10]}{' (truncated)' if len(seen_nodes) > 10 else ''}

    ## Available Nodes for Retrieve
    {available_nodes[:10] if available_nodes else 'None'}{' (truncated)' if len(available_nodes) > 10 else ''}

    ## Retrieved Data (max 3 shown)
    """ + "\n".join([
        f"- Node {nid}: {retrieved_data[nid]}"
        for nid in list(retrieved_data)[:3]
    ]) + ("\n...(truncated)" if len(retrieved_data) > 3 else "") + """

    ## Messages Received (max 3)
    """ + "\n".join([
        f"- From Node {m['from']}: {m.get('content_preview', '[no preview]')}"
        for m in messages[:3]
    ]) + ("\n...(truncated)" if len(messages) > 3 else "") + """

    ## Memory (max 2)
    """ + "\n".join([
        f"- Layer {m.get('layer', '?')} | Action={m.get('action')} | Label={m.get('label', '-')}"
        for m in memory[:2]
    ]) + ("\n...(truncated)" if len(memory) > 2 else "") + """

    ## Action Options (reply with JSON only)
    1. 📥 Retrieve    
    - Retrieve all nodes from {neighbors} list (integer IDs)
    - Syntax: {{ "action_type": "retrieve", "target_nodes": [Retrieve IDs], "info_type": "text" }}

    2. 📢 Broadcast
    - Broadcast message to all {neighbors} (integer IDs)
    - Syntax: {{ "action_type": "broadcast", "target_nodes": [Broadcast IDs], "message": "..." }}  

    3. 🎯 Update  
    - Syntax: {{ "action_type": "update", "predicted_label": label_id }}

    4. 💤 No Operation  
    - Syntax: {{ "action_type": "no_op" }}

    ### 🔍 Retrieve Tips
    - Choose from **Available Nodes for Retrieve**
    - Only use **integer node IDs**
    - Avoid any node in **Seen Nodes**

    Reply with:
    {"action_type": "...", "target_nodes": [...], ...}
    """

        return prompt

    def _format_layer_prompt(self, context: Dict[str, Any]) -> str:
        return f"""You are the controller for a graph neural network.\n\nThe network has completed layer {context['current_layer']} of processing.\n\nMax layers: {context['max_layers']}\nCurrent layer: {context['current_layer']}\n\nBased on the progress, decide whether to:\n1. Continue to the next layer\n2. End processing and output final results\n\nRespond with either \"continue\" or \"end\".\n"""

    def _parse_action(self, response: str) -> Dict[str, Any]:
        import re, json
        try:
            # 尝试从 ```json\n{...}\n``` 或纯 {...} 中提取
            code_blocks = re.findall(r"```json\s*({.*?})\s*```", response, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r"({.*?})", response, re.DOTALL)

            for block in code_blocks:
                try:
                    cleaned = re.sub(r"//.*", "", block)  # 去掉注释
                    parsed = json.loads(cleaned)

                    # 清洗 target_nodes
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
