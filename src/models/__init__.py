"""
Neural network models for drug discovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class MolecularFingerprintNet(nn.Module):
    """Neural network for molecular fingerprint-based prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x).squeeze(-1)


class MolecularGraphNet(nn.Module):
    """Graph Neural Network for molecular property prediction."""
    
    def __init__(
        self,
        node_features: int = 6,
        edge_features: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn"
    ):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            self.convs.append(conv)
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Output
        return self.output(x).squeeze(-1)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass with ensemble averaging."""
        predictions = []
        
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)
        
        # Weighted average
        predictions = torch.stack(predictions, dim=0)
        weights = self.weights.to(predictions.device).view(-1, 1)
        
        return torch.sum(predictions * weights, dim=0)


class UncertaintyModel(nn.Module):
    """Model with uncertainty estimation using Monte Carlo dropout."""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_samples: int = 10
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_samples = num_samples
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass with uncertainty estimation."""
        predictions = []
        
        # Enable dropout for uncertainty estimation
        self.base_model.train()
        
        for _ in range(self.num_samples):
            pred = self.base_model(*args, **kwargs)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and variance
        mean = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        
        return mean, variance
    
    def predict_with_uncertainty(self, *args, **kwargs) -> tuple:
        """Predict with uncertainty bounds."""
        mean, variance = self.forward(*args, **kwargs)
        std = torch.sqrt(variance)
        
        # 95% confidence interval
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        
        return mean, lower, upper


def create_model(
    model_type: str,
    input_dim: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: Type of model ("fingerprint", "graph", "ensemble")
        input_dim: Input dimension (required for fingerprint models)
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    if model_type == "fingerprint":
        if input_dim is None:
            raise ValueError("input_dim required for fingerprint model")
        return MolecularFingerprintNet(input_dim=input_dim, **kwargs)
    
    elif model_type == "graph":
        return MolecularGraphNet(**kwargs)
    
    elif model_type == "ensemble":
        models = kwargs.get("models", [])
        weights = kwargs.get("weights", None)
        return EnsembleModel(models, weights)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
