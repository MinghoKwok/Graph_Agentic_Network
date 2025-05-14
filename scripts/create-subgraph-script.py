"""
Script to create a pre-processed subgraph from a larger dataset

This is useful for faster experimentation with the full Graph Agentic Network.
"""

import torch
import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.dataset import load_or_create_dataset


def main():
    parser = argparse.ArgumentParser(description='Create subgraph dataset')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='Dataset name')
    parser.add_argument('--size', type=int, default=1000,
                        help='Subgraph size')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    args = parser.parse_args()
    
    # Load dataset and create subgraph
    print(f"Creating {args.size}-node subgraph from {args.dataset}...")
    subgraph = load_or_create_dataset(args.dataset, use_subgraph=True, subgraph_size=args.size)
    
    # Determine output path
    if args.output is None:
        output_path = os.path.join(config.DATA_DIR, f"{args.dataset}_subgraph_{args.size}.pt")
    else:
        output_path = args.output
    
    # Save subgraph
    torch.save(subgraph, output_path)
    print(f"Subgraph saved to {output_path}")


if __name__ == "__main__":
    main()
