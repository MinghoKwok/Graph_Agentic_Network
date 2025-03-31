"""
Node classification experiment for Graph Agentic Network
"""

import torch
import os
import json
import time
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import config
from gan.llm import LLMInterface, MockLLMInterface
from gan.graph import GraphAgenticNetwork
from gan.utils import seed_everything, evaluate_node_classification, compare_results
from data.dataset import load_or_create_dataset
from baselines.gcn import GCNBaseline


def run_node_classification(
    dataset_name: str = config.DATASET_NAME,
    use_subgraph: bool = True,
    subgraph_size: int = 1000,
    use_mock_llm: bool = False,
    num_layers: int = config.NUM_LAYERS,
    batch_size: Optional[int] = None,
    save_results: bool = True,
    result_dir: Optional[str] = None,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Run node classification experiment.
    
    Args:
        dataset_name: Name of the dataset to use
        use_subgraph: Whether to use a subgraph for faster experiments
        subgraph_size: Size of the subgraph
        use_mock_llm: Whether to use a mock LLM for testing
        num_layers: Number of GAN layers
        batch_size: Batch size for processing nodes
        save_results: Whether to save the results
        result_dir: Directory to save results
        visualize: Whether to visualize the results
        
    Returns:
        Dictionary of experiment results
    """
    # Set random seed
    seed_everything()
    
    # Set result directory
    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(config.RESULTS_DIR, f"node_classification_{timestamp}")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name} (use_subgraph={use_subgraph}, size={subgraph_size})")
    dataset = load_or_create_dataset(dataset_name, use_subgraph, subgraph_size)
    
    # Extract components
    adj_matrix = dataset['adj_matrix']
    node_features = dataset['node_features']
    labels = dataset['labels']
    train_idx = dataset['train_idx']
    val_idx = dataset['val_idx']
    test_idx = dataset['test_idx']
    num_classes = dataset['num_classes']
    
    # Initialize LLM interface
    config.LLM_BACKEND = "mock" if use_mock_llm else "remote"
    llm_interface = LLMInterface(model_name=config.LLM_MODEL)
    
    # Create GAN model
    print(f"Creating Graph Agentic Network with {num_layers} layers")
    gan = GraphAgenticNetwork(
        adj_matrix=adj_matrix,
        node_features=node_features,
        llm_interface=llm_interface,
        labels=labels,
        num_layers=num_layers
    )
    
    # Run GAN
    print("Running Graph Agentic Network")
    start_time = time.time()
    gan.forward(batch_size=batch_size)
    gan_time = time.time() - start_time
    print(f"GAN completed in {gan_time:.2f} seconds")
    
    # Get GAN predictions
    gan_predictions = gan.get_node_predictions()
    
    # Create a mask for nodes with predictions
    pred_mask = (gan_predictions > 0)
    
    # Evaluate GAN
    gan_metrics = {
        # Full set evaluation
        'all': evaluate_node_classification(gan_predictions, labels),
        # Train set evaluation
        'train': evaluate_node_classification(gan_predictions[train_idx], labels[train_idx]),
        # Validation set evaluation
        'val': evaluate_node_classification(gan_predictions[val_idx], labels[val_idx]),
        # Test set evaluation
        'test': evaluate_node_classification(gan_predictions[test_idx], labels[test_idx])
    }
    
    print("GAN Results:")
    print(f"  Train Accuracy: {gan_metrics['train']['accuracy']:.4f}")
    print(f"  Val Accuracy: {gan_metrics['val']['accuracy']:.4f}")
    print(f"  Test Accuracy: {gan_metrics['test']['accuracy']:.4f}")
    
    # Run baseline (GCN)
    print("\nRunning GCN baseline")
    
    # Extract edge_index for PyTorch Geometric
    edge_index = adj_matrix.nonzero().t().contiguous()
    
    # Initialize GCN
    gcn = GCNBaseline(
        in_channels=node_features.size(1),
        out_channels=num_classes
    )
    
    # Train GCN
    start_time = time.time()
    gcn.train(
        edge_index=edge_index,
        node_features=node_features,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=config.GCN_EPOCHS
    )
    gcn_time = time.time() - start_time
    print(f"GCN completed in {gcn_time:.2f} seconds")
    
    # Evaluate GCN
    gcn_metrics = {
        'train': gcn.evaluate(edge_index, node_features, labels, train_idx),
        'val': gcn.evaluate(edge_index, node_features, labels, val_idx),
        'test': gcn.evaluate(edge_index, node_features, labels, test_idx)
    }
    
    print("GCN Results:")
    print(f"  Train Accuracy: {gcn_metrics['train']['accuracy']:.4f}")
    print(f"  Val Accuracy: {gcn_metrics['val']['accuracy']:.4f}")
    print(f"  Test Accuracy: {gcn_metrics['test']['accuracy']:.4f}")
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'gan': {
            'metrics': gan_metrics,
            'time': gan_time,
            'num_layers': num_layers,
            'llm_model': config.LLM_MODEL if not use_mock_llm else 'mock',
            'batch_size': batch_size
        },
        'gcn': {
            'metrics': gcn_metrics,
            'time': gcn_time,
            'hidden_dim': config.GCN_HIDDEN_DIM,
            'num_layers': config.GCN_NUM_LAYERS,
            'epochs': config.GCN_EPOCHS
        }
    }
    
    # Save results
    if save_results:
        result_file = os.path.join(result_dir, "results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
        
        print(f"Results saved to {result_file}")
    
    # Visualize results
    if visualize:
        # Compare test set metrics
        compare_results(
            gan_metrics['test'],
            gcn_metrics['test'],
            title=f"Model Comparison on {dataset_name} (Test Set)",
            save_path=os.path.join(result_dir, "model_comparison.png") if save_results else None
        )
        
        # Plot training curves for GCN
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot([m['loss'] for m in gcn.train_metrics], label='Train Loss')
        if gcn.val_metrics:
            plt.plot([m['loss'] for m in gcn.val_metrics], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GCN Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot([m['acc'] for m in gcn.train_metrics], label='Train Acc')
        if gcn.val_metrics:
            plt.plot([m['acc'] for m in gcn.val_metrics], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('GCN Training Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_results:
            plt.savefig(os.path.join(result_dir, "gcn_training.png"))
        else:
            plt.show()
    
    return results


if __name__ == "__main__":
    # If run directly, execute the node classification experiment
    # For faster testing, use a subgraph and mock LLM
    run_node_classification(
        use_subgraph=False,
        subgraph_size=1000,
        use_mock_llm=False,  # Set to False to use actual LLM
        num_layers=1,
        batch_size=64
    )
