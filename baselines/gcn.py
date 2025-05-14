"""
GCN baseline implementation for comparison with Graph Agentic Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

import config
from gan.utils import seed_everything


class GCNModel(nn.Module):
    """Graph Convolutional Network model."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int = 2, dropout: float = 0.5):
        """
        Initialize GCN model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features (classes)
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features
            edge_index: Graph connectivity

        Returns:
            Node predictions
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class GCNBaseline:
    """GCN baseline wrapper for easier comparison."""

    def __init__(self, in_channels: int, hidden_channels: int = config.GCN_HIDDEN_DIM, 
                 out_channels: int = None, num_layers: int = config.GCN_NUM_LAYERS, 
                 dropout: float = config.GCN_DROPOUT, lr: float = config.GCN_LEARNING_RATE,
                 weight_decay: float = config.GCN_WEIGHT_DECAY, device: str = None):
        """
        Initialize GCN baseline.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features (classes)
            num_layers: Number of GCN layers
            dropout: Dropout probability
            lr: Learning rate
            weight_decay: Weight decay factor
            device: Device to use
        """
        seed_everything()

        self.device = device if device else config.DEVICE
        self.out_channels = out_channels

        # Defer model creation until we know out_channels
        self.model = None
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

    def _create_model(self):
        """Create the GCN model."""
        if self.model is None:
            if self.out_channels is None:
                raise ValueError("out_channels must be set before creating the model")

            self.model = GCNModel(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )

    def train(self, edge_index: torch.Tensor, node_features: torch.Tensor, 
              labels: torch.Tensor, train_idx: torch.Tensor, 
              val_idx: Optional[torch.Tensor] = None, 
              epochs: int = config.GCN_EPOCHS, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the GCN model.

        Args:
            edge_index: Graph edge indices
            node_features: Node features
            labels: Node labels
            train_idx: Training node indices
            val_idx: Validation node indices
            epochs: Number of epochs
            verbose: Whether to print progress

        Returns:
            Dictionary of training metrics
        """
        # Set out_channels if not already set
        if self.out_channels is None:
            self.out_channels = labels.max().item() + 1

        # Create model if not already created
        self._create_model()

        # Ensure data is on the correct device
        edge_index = edge_index.to(self.device)
        node_features = node_features.to(self.device)
        labels = labels.to(self.device)
        train_idx = train_idx.to(self.device)
        if val_idx is not None:
            val_idx = val_idx.to(self.device)

        # Training loop
        for epoch in tqdm(range(epochs), desc="Training GCN", disable=not verbose):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()

            out = self.model(node_features, edge_index)
            loss = F.cross_entropy(out[train_idx], labels[train_idx])

            loss.backward()
            self.optimizer.step()

            # Compute training metrics
            train_acc = self._compute_accuracy(out[train_idx], labels[train_idx])

            epoch_metrics = {
                'loss': loss.item(),
                'acc': train_acc
            }
            self.train_metrics.append(epoch_metrics)

            # Validation if provided
            if val_idx is not None:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(node_features, edge_index)
                    val_loss = F.cross_entropy(out[val_idx], labels[val_idx])
                    val_acc = self._compute_accuracy(out[val_idx], labels[val_idx])

                    val_metrics = {
                        'loss': val_loss.item(),
                        'acc': val_acc
                    }
                    self.val_metrics.append(val_metrics)

            # Print progress
            if verbose and (epoch + 1) % config.LOG_INTERVAL == 0:
                log_msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}"
                if val_idx is not None:
                    log_msg += f", Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}"
                print(log_msg)

        return {
            'train': self.train_metrics,
            'val': self.val_metrics
        }

    def predict(self, edge_index: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the GCN model.

        Args:
            edge_index: Graph edge indices
            node_features: Node features

        Returns:
            Predicted class indices
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        edge_index = edge_index.to(self.device)
        node_features = node_features.to(self.device)

        with torch.no_grad():
            out = self.model(node_features, edge_index)
            pred = out.argmax(dim=1)

        return pred

    def evaluate(self, edge_index: torch.Tensor, node_features: torch.Tensor, 
                labels: torch.Tensor, idx: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the GCN model.

        Args:
            edge_index: Graph edge indices
            node_features: Node features
            labels: Node labels
            idx: Node indices to evaluate on

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        edge_index = edge_index.to(self.device)
        node_features = node_features.to(self.device)
        labels = labels.to(self.device)
        idx = idx.to(self.device)

        with torch.no_grad():
            out = self.model(node_features, edge_index)
            loss = F.cross_entropy(out[idx], labels[idx])
            acc = self._compute_accuracy(out[idx], labels[idx])

            # Calculate F1 scores
            pred = out[idx].argmax(dim=1).cpu()
            labels_cpu = labels[idx].cpu()

            from sklearn.metrics import f1_score
            f1_micro = f1_score(labels_cpu, pred, average='micro')
            f1_macro = f1_score(labels_cpu, pred, average='macro')

        return {
            'loss': loss.item(),
            'accuracy': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }

    def _compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute classification accuracy.

        Args:
            logits: Prediction logits
            labels: Ground truth labels

        Returns:
            Accuracy value
        """
        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()
        return correct / labels.size(0)


# ✅ Add this for link prediction compatibility
GCN = GCNModel
