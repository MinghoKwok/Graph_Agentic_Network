"""
LLM interface for decision making in the Graph Agentic Network
"""

import json
import torch
import os
import re
import random
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

import config
from config import DEBUG_LLM


class LLMInterface: 
    """Interface for communicating with the LLM for agent decision making."""
    
    def __init__(self, model_name: str = config.LLM_MODEL, device: Optional[str] = None):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name or path of the LLM model
            device: Device to use for computation
        """
        self.model_name = model_name
        self.device = device if device else config.DEVICE
        
        # Track if we've initialized the model
        self.initialized = False
        
        # Load tokenizer immediately, but delay model loading until needed
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LOCAL_LLM_PATH if config.USE_LOCAL_LLM else model_name
        )
    
    def _initialize_model(self):
        """Initialize the LLM model if not already initialized."""
        if not self.initialized:
            print(f"Initializing LLM model: {self.model_name}")
            
            # Load model
            model_path = config.LOCAL_LLM_PATH if config.USE_LOCAL_LLM else self.model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
                device_map="auto" if config.MULTI_GPU else self.device
            )
            
            self.initialized = True
    
    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide an action based on the node's context.
        
        Args:
            context: Context dictionary containing node state and environment info
            
        Returns:
            Decision dictionary specifying the action to take
        """
        # Format the prompt
        prompt = self._format_action_prompt(context)

        # Print the prompt for debugging
        if DEBUG_LLM:
            print(f"\n================ LLM Prompt for Node {context['node_id']} Layer {context['layer']} ================\n")
            print(prompt)
            print("============================================================================\n")

        
        # Generate response from LLM
        response = self.generate_response(prompt)

        # Debug: print raw LLM response
        if DEBUG_LLM:
            print(f"\n🧠 LLM raw response:\n{response}\n")

        
        # Parse the action from the response
        action = self._parse_action(response)

        # Debug: print parsed action
        if DEBUG_LLM:
            print(f"✅ Parsed action:\n{action}\n")
        
        return action
    
    def determine_next_layer(self, context: Dict[str, Any]) -> bool:
        """
        Determine whether to continue to the next layer.
        
        Args:
            context: Context dictionary containing network state
            
        Returns:
            Boolean indicating whether to continue to the next layer
        """
        # Format the prompt
        prompt = self._format_layer_prompt(context)
        
        # Generate response from LLM
        response = self.generate_response(prompt)
        
        # Parse the response
        continue_next_layer = "continue" in response.lower()
        
        return continue_next_layer
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated response
        """
        # Initialize model if needed
        self._initialize_model()
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )
        
        # Decode response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the part after the prompt
        response = response[len(prompt):]
        
        return response
    
    def _format_action_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format the prompt for action decisions.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted prompt
        """
        node_id = context["node_id"]
        layer = context["layer"]
        features = context["features"]
        neighbors = context["neighbors"][:10] if len(context["neighbors"]) > 10 else context["neighbors"]
        total_neighbors = context["total_neighbors"]
        messages = context["messages"]
        retrieved_data = context.get("retrieved_data", {})
        
        prompt = f"""You are an intelligent node agent (Node {node_id}) in a graph neural network.
Based on your current state and observations, you need to decide what action to take.

Current Layer: {layer}
Your Features: {features}
Your Neighbors: {neighbors}{' (truncated)' if total_neighbors > len(neighbors) else ''}
Total Neighbors: {total_neighbors}

"""
        
        # Include retrieved data if available
        if retrieved_data:
            prompt += "Retrieved data from previous actions:\n"
            preview_count = min(3, len(retrieved_data))
            previewed_items = list(retrieved_data.items())[:preview_count]
            
            for neighbor_id, data in previewed_items:
                prompt += f"- Node {neighbor_id}: {data}\n"
            
            if len(retrieved_data) > preview_count:
                prompt += f"(and {len(retrieved_data) - preview_count} more nodes)\n"
            
            prompt += "\n"
        
        # Include recent messages if available
        if messages:
            prompt += "Recent messages:\n"
            for msg in messages:
                prompt += f"- From Node {msg['from']} (Layer {msg['layer']}): content preview {msg['content_preview']}\n"
            prompt += "\n"
        
        # Action menu
        prompt += """Available actions:
1. retrieve - Retrieve information from neighbors
   Example: {"action_type": "retrieve", "target_nodes": [1, 2, 3], "info_type": "features"}
   - info_type can be "features", "label", or "both"

2. broadcast - Send a message to neighbors
   Example: {"action_type": "broadcast", "target_nodes": [1, 2, 3], "message": [0.5, 0.3, 0.7]}

3. update - Update your own state
   Example: {"action_type": "update", "predicted_label": 5}

4. no_op - Do nothing
   Example: {"action_type": "no_op"}

Choose the action that best fits your current situation and goals. Respond with a valid JSON object.

"""
        
        return prompt
    
    def _format_layer_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format the prompt for layer decisions.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted prompt
        """
        current_layer = context["current_layer"]
        max_layers = context["max_layers"]
        
        prompt = f"""You are the controller for a graph neural network.
The network has completed layer {current_layer} of processing.

Max layers: {max_layers}
Current layer: {current_layer}

Based on the progress, decide whether to:
1. Continue to the next layer
2. End processing and output final results

Respond with either "continue" or "end".

"""
        
        return prompt
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """
        Parse an action from the LLM response.
        
        Args:
            response: The LLM response text
            
        Returns:
            Parsed action dictionary
        """
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(response)
            action = json.loads(json_str)
            return action
        except Exception as e:
            # Fallback to a simple action if parsing fails
            print(f"Failed to parse action: {e}")
            print(f"Response was: {response}")
            return {"action_type": "no_op"}
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string
        """
        # Look for JSON between triple backticks
        if "```json" in text and "```" in text:
            start_idx = text.find("```json") + 7
            end_idx = text.find("```", start_idx)
            if end_idx > start_idx:
                return text[start_idx:end_idx].strip()
        
        # Look for JSON between backticks
        if "```" in text:
            start_idx = text.find("```") + 3
            end_idx = text.find("```", start_idx)
            if end_idx > start_idx:
                return text[start_idx:end_idx].strip()
        
        # Look for JSON between { and }
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx+1].strip()
        
        # Fallback: return empty JSON
        return "{}"


class MockLLMInterface(LLMInterface):
    """A mock LLM interface for testing without actual LLM calls."""
    
    def __init__(self):
        """Initialize the mock interface."""
        self.initialized = True

    def generate_response(self, prompt: str) -> str:
        """
        Generate a mock response.
        
        Args:
            prompt: The input prompt
            
        Returns:
            A mock response
        """
        if "action" in prompt.lower():
            if "Current Layer: 0" in prompt:
                # 🧠 从 prompt 中提取邻居节点
                neighbor_match = re.search(r"Your Neighbors: \[(.*?)\]", prompt)
                if neighbor_match:
                    neighbor_str = neighbor_match.group(1)
                    neighbors = [int(n.strip()) for n in neighbor_str.split(",") if n.strip().isdigit()]
                    target_nodes = random.sample(neighbors, min(3, len(neighbors))) if neighbors else []
                    
                    return json.dumps({
                        "action_type": "retrieve",
                        "target_nodes": target_nodes,
                        "info_type": "features"
                    })
                else:
                    return '{"action_type": "no_op"}'
            else:
                # 后续层做 update
                return json.dumps({
                    "action_type": "update",
                    "predicted_label": random.randint(0, 39)  # 假设有 40 类
                })
        
        elif "layer" in prompt.lower():
            return "continue"
        
        return "no_op"
