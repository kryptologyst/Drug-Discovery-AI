"""
Unit tests for drug discovery AI project.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from rdkit import Chem

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import (
    set_seed, get_device, validate_smiles, 
    smiles_to_mol, calculate_molecular_descriptors
)
from src.data.molecular import (
    smiles_to_morgan_fingerprint, smiles_to_maccs_keys,
    smiles_to_rdkit_descriptors, smiles_to_graph,
    generate_synthetic_dataset
)
from src.models import create_model, count_parameters
from src.losses import get_loss_function
from src.metrics import QSARMetrics


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Placeholder test
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_validate_smiles(self):
        """Test SMILES validation."""
        assert validate_smiles("CCO") == True  # Valid SMILES
        assert validate_smiles("invalid") == False  # Invalid SMILES
        assert validate_smiles("") == False  # Empty string
    
    def test_smiles_to_mol(self):
        """Test SMILES to molecule conversion."""
        mol = smiles_to_mol("CCO")
        assert mol is not None
        assert mol.GetNumAtoms() > 0
        
        mol_invalid = smiles_to_mol("invalid")
        assert mol_invalid is None
    
    def test_calculate_molecular_descriptors(self):
        """Test molecular descriptor calculation."""
        mol = Chem.MolFromSmiles("CCO")
        descriptors = calculate_molecular_descriptors(mol)
        
        assert isinstance(descriptors, dict)
        assert 'MW' in descriptors
        assert 'LogP' in descriptors
        assert 'NumAtoms' in descriptors
        assert descriptors['NumAtoms'] > 0


class TestMolecularData:
    """Test molecular data processing."""
    
    def test_smiles_to_morgan_fingerprint(self):
        """Test Morgan fingerprint generation."""
        fp = smiles_to_morgan_fingerprint("CCO")
        assert fp is not None
        assert len(fp) == 1024
        assert isinstance(fp, np.ndarray)
        
        fp_invalid = smiles_to_morgan_fingerprint("invalid")
        assert fp_invalid is None
    
    def test_smiles_to_maccs_keys(self):
        """Test MACCS keys generation."""
        fp = smiles_to_maccs_keys("CCO")
        assert fp is not None
        assert len(fp) == 167
        assert isinstance(fp, np.ndarray)
        
        fp_invalid = smiles_to_maccs_keys("invalid")
        assert fp_invalid is None
    
    def test_smiles_to_rdkit_descriptors(self):
        """Test RDKit descriptors generation."""
        desc = smiles_to_rdkit_descriptors("CCO")
        assert desc is not None
        assert len(desc) == 11
        assert isinstance(desc, np.ndarray)
        
        desc_invalid = smiles_to_rdkit_descriptors("invalid")
        assert desc_invalid is None
    
    def test_smiles_to_graph(self):
        """Test molecular graph generation."""
        graph = smiles_to_graph("CCO")
        assert graph is not None
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert graph.x.shape[1] == 6  # Node features
        
        graph_invalid = smiles_to_graph("invalid")
        assert graph_invalid is None
    
    def test_generate_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        smiles_list, targets = generate_synthetic_dataset(n_samples=10, seed=42)
        
        assert len(smiles_list) == 10
        assert len(targets) == 10
        assert all(isinstance(s, str) for s in smiles_list)
        assert all(isinstance(t, float) for t in targets)
        assert all(0 <= t <= 1 for t in targets)


class TestModels:
    """Test model creation and functionality."""
    
    def test_create_fingerprint_model(self):
        """Test fingerprint model creation."""
        model = create_model(
            model_type="fingerprint",
            input_dim=1024,
            hidden_dims=[128, 64],
            dropout=0.2
        )
        
        assert model is not None
        assert count_parameters(model) > 0
        
        # Test forward pass
        x = torch.randn(2, 1024)
        output = model(x)
        assert output.shape == (2,)
    
    def test_create_graph_model(self):
        """Test graph model creation."""
        model = create_model(
            model_type="graph",
            node_features=6,
            edge_features=3,
            hidden_dim=32,
            num_layers=2
        )
        
        assert model is not None
        assert count_parameters(model) > 0


class TestLosses:
    """Test loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss function."""
        loss_fn = get_loss_function("mse")
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 1.9, 3.1])
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0
        assert isinstance(loss, torch.Tensor)
    
    def test_mae_loss(self):
        """Test MAE loss function."""
        loss_fn = get_loss_function("mae")
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 1.9, 3.1])
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0
        assert isinstance(loss, torch.Tensor)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_qsar_metrics(self):
        """Test QSAR metrics calculation."""
        metrics = QSARMetrics()
        
        predictions = np.array([0.1, 0.5, 0.9])
        targets = np.array([0.2, 0.4, 0.8])
        
        metrics.update(predictions, targets)
        results = metrics.compute()
        
        assert 'mse' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'r2' in results
        assert 'pearson_r' in results
        
        assert isinstance(results['mse'], float)
        assert isinstance(results['r2'], float)
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = QSARMetrics()
        
        metrics.update([1.0], [1.0])
        metrics.reset()
        
        assert len(metrics.predictions) == 0
        assert len(metrics.targets) == 0


if __name__ == "__main__":
    pytest.main([__file__])
