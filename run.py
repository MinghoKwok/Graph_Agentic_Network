"""
Main script for running Graph Agentic Network experiments
"""

import argparse
import torch
import sys
import os

import config
from experiments.run_node_classification import run_node_classification


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Graph Agentic Network')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME,
                        help='Dataset name (default: ogbn-arxiv)')
    parser.add_argument('--use-subgraph', action='store_true',
                        help='Use a subgraph of the dataset for faster experiments')
    parser.add_argument('--subgraph-size', type=int, default=1000,
                        help='Size of the subgraph to use if use-subgraph is enabled')
    
    # Model options
    parser.add_argument('--num-layers', type=int, default=config.NUM_LAYERS,
                        help='Number of GAN layers to run')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for processing nodes (None for all at once)')
    parser.add_argument('--use-mock-llm', action='store_true',
                        help='Use a mock LLM for testing')
    
    # Output options
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Do not visualize results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: auto-generated)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Print available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available, using CPU")
    
    # Run experiment
    results = run_node_classification(
        dataset_name=args.dataset,
        use_subgraph=args.use_subgraph,
        subgraph_size=args.subgraph_size,
        use_mock_llm=args.use_mock_llm,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        save_results=not args.no_save,
        result_dir=args.output_dir,
        visualize=not args.no_visualize
    )
    
    return results


if __name__ == "__main__":
    main()
