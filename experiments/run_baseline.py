import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import time


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    metrics = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f"{split}_mask")
        y_true = data.y[mask]
        y_pred = pred[mask]
        acc = (y_pred == y_true).float().mean().item()
        f1_micro = f1_score(y_true.cpu(), y_pred.cpu(), average='micro')
        f1_macro = f1_score(y_true.cpu(), y_pred.cpu(), average='macro')
        metrics[split] = {
            'accuracy': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }
    return metrics


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 8, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def run_baselines_on_planetoid(dataset_name='Cora', epochs=200, hidden_dim=16, dropout=0.5, lr=0.01):
    data = Planetoid(root='data', name=dataset_name)[0]
    in_dim = data.num_node_features
    out_dim = data.y.max().item() + 1

    models = {
        'GCN': GCN(in_dim, hidden_dim, out_dim, dropout),
        'GAT': GAT(in_dim, hidden_dim, out_dim, dropout),
        'GraphSAGE': GraphSAGE(in_dim, hidden_dim, out_dim, dropout),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        t0 = time.time()
        for epoch in range(epochs):
            train(model, data, optimizer)
        elapsed = time.time() - t0
        metrics = evaluate(model, data)
        results[name] = {
            'metrics': metrics,
            'time': elapsed
        }
        print(f"{name} - Test Accuracy: {metrics['test']['accuracy']:.4f} - Time: {elapsed:.2f}s")

    return results


if __name__ == '__main__':
    run_baselines_on_planetoid('Cora')
