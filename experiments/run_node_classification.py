"""
Node classification experiment for Graph Agentic Network
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

import config
from gan.llm import LLMInterface
from gan.graph import GraphAgenticNetwork
from gan.utils import seed_everything, evaluate_node_classification
from data.dataset import load_or_create_dataset
from baselines.gcn import GCNBaseline
from baselines.gat import GAT
from baselines.graphsage import GraphSAGE
import csv

def run_node_classification(
    dataset_name: str = config.DATASET_NAME,
    use_subgraph: bool = True,
    subgraph_size: int = 1000,
    use_mock_llm: bool = False,
    num_layers: int = config.NUM_LAYERS,
    batch_size: Optional[int] = None,
    run_gan: bool = True,
    run_baselines: bool = True,
    save_results: bool = True,
    result_dir: Optional[str] = None,
    visualize: bool = True
) -> Dict[str, Any]:
    seed_everything()

    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(config.RESULTS_DIR, f"node_classification_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_name} (use_subgraph={use_subgraph}, size={subgraph_size})")
    dataset = load_or_create_dataset(dataset_name, use_subgraph, subgraph_size)

    adj_matrix = dataset['adj_matrix']
    node_features = dataset['node_features']
    labels = dataset['labels']
    train_idx = dataset['train_idx']
    val_idx = dataset['val_idx']
    test_idx = dataset['test_idx']
    num_classes = dataset['num_classes']

    # 加载文本形式的节点描述
    # with open(f"../data/{dataset_name}/cora_text_graph_simplified.jsonl") as f:
    #     node_texts = {int(json.loads(line)["node_id"]): json.loads(line)["text"] for line in f}
    node_texts = dataset['node_texts']
    print(f"Loaded {len(node_texts)} node texts for GAN.")

    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat()
    }

    if run_gan:
        from gan.llm import MockLLMInterface, RemoteLLMInterface
        config.LLM_BACKEND = "mock" if use_mock_llm else "remote"
        llm_interface = MockLLMInterface() if use_mock_llm else LLMInterface(model_name=config.LLM_MODEL)

        print(f"⚙️  GAN config: LLM={config.LLM_MODEL}, Layers={num_layers}, Batch size={batch_size}, Backend={config.LLM_BACKEND}")

        print(f"Creating Graph Agentic Network with {num_layers} layers")
        gan = GraphAgenticNetwork(
            adj_matrix=adj_matrix,
            node_texts=node_texts,
            llm_interface=llm_interface,
            labels=labels,
            num_layers=num_layers,
            train_idx=train_idx
        )

        print("Running Graph Agentic Network")
        start_time = time.time()
        gan.forward(batch_size=batch_size)
        gan_time = time.time() - start_time
        print(f"GAN completed in {gan_time:.2f} seconds")

        gan_predictions = gan.get_node_predictions()

        # 记录测试节点的预测结果
        test_results = []
        for idx in test_idx:
            node_id = int(idx)  # 转换为整数
            node = gan.graph.nodes[node_id]  # 使用整数索引
            test_results.append({
                'node_id': node_id,
                'predicted_label': int(gan_predictions[idx]),
                'true_label': int(labels[idx]),
                'memory': [str(m) for m in node.state.memory] if node.state.memory else []
            })

        # 保存到JSON文件
        test_results_file = os.path.join(result_dir, "test_predictions.json")
        with open(test_results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"✅ Test predictions saved to {test_results_file}")

        gan_metrics = {
            'all': evaluate_node_classification(gan_predictions, labels),
            'train': evaluate_node_classification(gan_predictions[train_idx], labels[train_idx]),
            'val': evaluate_node_classification(gan_predictions[val_idx], labels[val_idx]),
            'test': evaluate_node_classification(gan_predictions[test_idx], labels[test_idx])
        }

        print("GAN Results:")
        print(f"  Train Accuracy: {gan_metrics['train']['accuracy']:.4f}")
        print(f"  Val Accuracy: {gan_metrics['val']['accuracy']:.4f}")
        print(f"  Test Accuracy: {gan_metrics['test']['accuracy']:.4f}")

        results['gan'] = {
            'metrics': gan_metrics,
            'time': gan_time,
            'num_layers': num_layers,
            'llm_model': config.LLM_MODEL if not use_mock_llm else 'mock',
            'batch_size': batch_size
        }

    if run_baselines:
        edge_index = adj_matrix.nonzero().t().contiguous()
        baseline_results = {}

        for model_name in config.BASELINE_TYPES:
            print(f"\nRunning {model_name} baseline")
            if model_name == "GCN":
                model = GCNBaseline(node_features.size(1), num_classes)
                hidden_dim = config.GCN_HIDDEN_DIM
                dropout = config.GCN_DROPOUT
                lr = config.GCN_LEARNING_RATE
                weight_decay = config.GCN_WEIGHT_DECAY
                epochs = config.GCN_EPOCHS
            elif model_name == "GAT":
                model = GAT(node_features.size(1), config.GAT_HIDDEN_DIM, num_classes, dropout=config.GAT_DROPOUT).to(config.DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.GAT_LEARNING_RATE, weight_decay=config.GAT_WEIGHT_DECAY)
                epochs = config.GAT_EPOCHS
            elif model_name == "GraphSAGE":
                model = GraphSAGE(node_features.size(1), config.SAGE_HIDDEN_DIM, num_classes, dropout=config.SAGE_DROPOUT).to(config.DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.SAGE_LEARNING_RATE, weight_decay=config.SAGE_WEIGHT_DECAY)
                epochs = config.SAGE_EPOCHS

            start_time = time.time()

            if model_name == "GCN":
                model.train(edge_index, node_features, labels, train_idx, val_idx, epochs)
                metrics = {
                    'train': model.evaluate(edge_index, node_features, labels, train_idx),
                    'val': model.evaluate(edge_index, node_features, labels, val_idx),
                    'test': model.evaluate(edge_index, node_features, labels, test_idx),
                }
            else:
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    out = model(node_features.to(config.DEVICE), edge_index.to(config.DEVICE))
                    loss = torch.nn.functional.cross_entropy(out[train_idx], labels[train_idx].to(config.DEVICE))
                    loss.backward()
                    optimizer.step()

                model.eval()
                out = model(node_features.to(config.DEVICE), edge_index.to(config.DEVICE))
                from sklearn.metrics import f1_score
                metrics = {}
                for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
                    pred = out[idx].argmax(dim=1)
                    acc = (pred == labels[idx].to(config.DEVICE)).float().mean().item()
                    f1_micro = f1_score(labels[idx].cpu(), pred.cpu(), average='micro')
                    f1_macro = f1_score(labels[idx].cpu(), pred.cpu(), average='macro')
                    metrics[split] = {
                        'accuracy': acc,
                        'f1_micro': f1_micro,
                        'f1_macro': f1_macro
                    }

            elapsed = time.time() - start_time
            baseline_results[model_name.lower()] = {
                'metrics': metrics,
                'time': elapsed
            }
            print(f"{model_name} completed in {elapsed:.2f} seconds")

        results.update(baseline_results)

    if save_results:
        result_file = os.path.join(result_dir, "results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
        print(f"Results saved to {result_file}")
        csv_path = os.path.join(result_dir, "summary.csv")
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'split', 'accuracy', 'f1_micro', 'f1_macro'])
            for model in ['gan'] + [m.lower() for m in config.BASELINE_TYPES]:
                if model in results:
                    for split in ['train', 'val', 'test']:
                        m = results[model]['metrics'][split]
                        writer.writerow([
                            model,
                            split,
                            f"{m.get('accuracy', 0):.4f}",
                            f"{m.get('f1_micro', 0):.4f}",
                            f"{m.get('f1_macro', 0):.4f}"
                        ])
        print(f"Results CSV saved to {csv_path}")


    return results


if __name__ == "__main__":
    run_node_classification(
        use_subgraph=False,
        subgraph_size=100,
        use_mock_llm=False,
        num_layers=config.NUM_LAYERS,
        batch_size=64,
        run_gan=True,
        run_baselines=True
    )
