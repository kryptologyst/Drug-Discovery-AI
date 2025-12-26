"""
Molecular data processing utilities for drug discovery.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import networkx as nx


def smiles_to_morgan_fingerprint(
    smiles: str, 
    radius: int = 2, 
    n_bits: int = 1024
) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint.
    
    Args:
        smiles: SMILES string
        radius: Fingerprint radius
        n_bits: Number of bits in fingerprint
        
    Returns:
        Morgan fingerprint as numpy array or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def smiles_to_maccs_keys(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to MACCS keys.
    
    Args:
        smiles: SMILES string
        
    Returns:
        MACCS keys as numpy array or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return np.array(fp)


def smiles_to_rdkit_descriptors(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to RDKit molecular descriptors.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Molecular descriptors as numpy array or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        rdMolDescriptors.CalcNumRings(mol),
    ]
    
    return np.array(descriptors)


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric graph.
    
    Args:
        smiles: SMILES string
        
    Returns:
        PyTorch Geometric Data object or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
        ]
        atom_features.append(features)
    
    # Get edge indices and features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_indices.extend([[i, j], [j, i]])  # Undirected graph
        
        bond_features = [
            bond.GetBondType().real,
            bond.GetIsAromatic(),
            bond.GetIsConjugated(),
        ]
        edge_features.extend([bond_features, bond_features])
    
    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def batch_graphs(graphs: List[Data]) -> Batch:
    """Batch multiple graphs into a single batch.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        
    Returns:
        Batched graph data
    """
    return Batch.from_data_list(graphs)


def create_molecular_dataset(
    smiles_list: List[str],
    targets: Optional[List[float]] = None,
    fingerprint_type: str = "morgan",
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Create molecular dataset from SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        targets: Optional list of target values
        fingerprint_type: Type of fingerprint ("morgan", "maccs", "descriptors")
        **kwargs: Additional arguments for fingerprint generation
        
    Returns:
        Tuple of (features, targets)
    """
    features = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        if fingerprint_type == "morgan":
            fp = smiles_to_morgan_fingerprint(smiles, **kwargs)
        elif fingerprint_type == "maccs":
            fp = smiles_to_maccs_keys(smiles)
        elif fingerprint_type == "descriptors":
            fp = smiles_to_rdkit_descriptors(smiles)
        else:
            raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")
        
        if fp is not None:
            features.append(fp)
            valid_indices.append(i)
    
    features = np.array(features)
    
    if targets is not None:
        targets = np.array([targets[i] for i in valid_indices])
        return features, targets
    
    return features, None


def generate_synthetic_dataset(
    n_samples: int = 1000,
    seed: int = 42
) -> Tuple[List[str], List[float]]:
    """Generate synthetic molecular dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        Tuple of (smiles_list, targets)
    """
    np.random.seed(seed)
    
    # Common drug-like fragments
    fragments = [
        "CCO",           # ethanol
        "CCCC",          # butane
        "CC(=O)O",       # acetic acid
        "c1ccccc1",      # benzene
        "CCN(CC)CC",     # triethylamine
        "CC(C)O",        # isopropanol
        "C1=CC=CC=C1O",  # phenol
        "COC(=O)C",      # methyl acetate
        "CN(C)C=O",      # dimethylformamide
        "CC(C)CC(=O)O",  # valeric acid
        "CCN",           # ethylamine
        "CC(=O)N",       # acetamide
        "CCOC",          # ethyl methyl ether
        "CC(C)(C)O",     # tert-butanol
        "C1CCCCC1",      # cyclohexane
    ]
    
    smiles_list = []
    targets = []
    
    for _ in range(n_samples):
        # Randomly select fragments and combine
        n_fragments = np.random.randint(1, 4)
        selected_fragments = np.random.choice(fragments, n_fragments, replace=True)
        
        # Simple combination (in real scenario, would use proper chemistry)
        smiles = "".join(selected_fragments)
        
        # Generate synthetic target based on molecular properties
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Simple activity score based on molecular weight and logP
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            # Normalize and combine
            activity = (mw / 500.0) * 0.5 + (logp + 2) / 4 * 0.5
            activity = np.clip(activity + np.random.normal(0, 0.1), 0, 1)
            
            smiles_list.append(smiles)
            targets.append(activity)
    
    return smiles_list, targets
