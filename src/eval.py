"""
Main evaluation script for drug discovery models.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import set_seed, get_device, setup_logging, load_config
from src.data.molecular import create_molecular_dataset, generate_synthetic_dataset
from src.models import create_model
from src.eval import evaluate_model, compare_models, create_leaderboard


def load_model_and_data(config: DictConfig) -> tuple[nn.Module, DataLoader]:
    """Load trained model and test data."""
    
    # Generate test data (same as training for consistency)
    smiles_list, targets = generate_synthetic_dataset(
        n_samples=config.data.n_samples,
        seed=config.seed
    )
    
    # Create molecular features
    features, targets = create_molecular_dataset(
        smiles_list=smiles_list,
        targets=targets,
        fingerprint_type=config.data.fingerprint_type,
        radius=config.data.get('radius', 2),
        n_bits=config.data.get('n_bits', 1024)
    )
    
    # Create DataFrame for splitting
    df = pd.DataFrame({
        'features': [f.tolist() for f in features],
        'targets': targets
    })
    
    # Split data (same as training)
    from src.utils.core import create_train_val_test_split
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.seed
    )
    
    # Convert test data to tensors
    test_features = np.array([np.array(f) for f in test_df['features']])
    test_targets = test_df['targets'].values
    
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)
    
    # Create test dataset and loader
    test_dataset = TensorDataset(test_features, test_targets)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # Create model
    input_dim = test_features.shape[1]
    model = create_model(
        model_type=config.model.type,
        input_dim=input_dim,
        **config.model.params
    )
    
    # Load trained weights
    checkpoint_path = os.path.join(config.model.checkpoint_path, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, test_loader


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate drug discovery model")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Config file path")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs", help="Output directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logging
    logger = setup_logging(
        log_level=config.get('log_level', 'INFO'),
        log_file=os.path.join(args.output_dir, 'eval.log')
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting drug discovery model evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Update config with model path
    config.model.checkpoint_path = args.model_path
    
    # Load model and data
    model, test_loader = load_model_and_data(config)
    
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=get_device(),
        save_dir=args.output_dir
    )
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Test R²: {results['r2']:.4f}")
    logger.info(f"Test RMSE: {results['rmse']:.4f}")
    logger.info(f"Test MAE: {results['mae']:.4f}")
    logger.info(f"Pearson correlation: {results['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
