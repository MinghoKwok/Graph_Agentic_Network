"""
Script to test the connection to LLM models and ensure they're working properly
"""

import os
import sys
import argparse
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from gan.llm import LLMInterface


def main():
    parser = argparse.ArgumentParser(description='Test LLM connection')
    parser.add_argument('--model', type=str, default=config.LLM_MODEL,
                        help='LLM model to test')
    parser.add_argument('--prompt', type=str, 
                        default="You are a node in a graph neural network. What would you do with your neighboring information?",
                        help='Test prompt to send to the LLM')
    args = parser.parse_args()
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Initialize LLM interface
    print(f"\nInitializing LLM interface with model: {args.model}")
    llm = LLMInterface(model_name=args.model)
    
    # Test with a sample prompt
    print("\nSending test prompt to LLM...")
    response = llm.generate_response(args.prompt)
    
    print("\nLLM Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    print("\nConnection test complete!")


if __name__ == "__main__":
    main()
