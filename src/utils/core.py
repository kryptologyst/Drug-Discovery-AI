"""
Core utilities for drug discovery AI project.
"""

import os
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file."""
    OmegaConf.save(config, config_path)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES to RDKit molecule object."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def calculate_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate molecular descriptors for a molecule."""
    descriptors = {}
    
    # Basic descriptors
    descriptors['MW'] = Descriptors.MolWt(mol)
    descriptors['LogP'] = Descriptors.MolLogP(mol)
    descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
    descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    descriptors['TPSA'] = Descriptors.TPSA(mol)
    descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    
    # Additional descriptors
    descriptors['NumAtoms'] = mol.GetNumAtoms()
    descriptors['NumBonds'] = mol.GetNumBonds()
    descriptors['NumRings'] = rdMolDescriptors.CalcNumRings(mol)
    
    return descriptors


def early_stopping(
    val_losses: List[float], 
    patience: int = 10, 
    min_delta: float = 1e-4
) -> bool:
    """Check if early stopping criteria is met."""
    if len(val_losses) < patience + 1:
        return False
    
    best_loss = min(val_losses[:-patience])
    current_loss = val_losses[-1]
    
    return current_loss - best_loss > min_delta


def create_train_val_test_split(
    data: pd.DataFrame, 
    test_size: float = 0.2, 
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits."""
    # First split: train+val vs test
    train_val, test = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    return train, val, test


def normalize_features(
    train_features: np.ndarray,
    val_features: Optional[np.ndarray] = None,
    test_features: Optional[np.ndarray] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """Normalize features using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        train_features_norm = scaler.fit_transform(train_features)
    else:
        train_features_norm = scaler.transform(train_features)
    
    val_features_norm = None
    if val_features is not None:
        val_features_norm = scaler.transform(val_features)
    
    test_features_norm = None
    if test_features is not None:
        test_features_norm = scaler.transform(test_features)
    
    return train_features_norm, val_features_norm, test_features_norm, scaler


def suppress_warnings() -> None:
    """Suppress common warnings."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
