"""
Main training script for drug discovery models.
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
from src.train import Trainer, create_optimizer, create_scheduler
from src.eval import evaluate_model


def create_data_loaders(
    config: DictConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    
    # Generate or load data
    if config.data.use_synthetic:
        smiles_list, targets = generate_synthetic_dataset(
            n_samples=config.data.n_samples,
            seed=config.seed
        )
    else:
        # Load real data (placeholder for now)
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
    
    # Split data
    from src.utils.core import create_train_val_test_split
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.seed
    )
    
    # Convert to tensors
    def df_to_tensors(df):
        features = np.array([np.array(f) for f in df['features']])
        targets = df['targets'].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    train_features, train_targets = df_to_tensors(train_df)
    val_features, val_targets = df_to_tensors(val_df)
    test_features, test_targets = df_to_tensors(test_df)
    
    # Create datasets
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    test_dataset = TensorDataset(test_features, test_targets)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train drug discovery model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logging
    logger = setup_logging(
        log_level=config.get('log_level', 'INFO'),
        log_file=os.path.join(args.output_dir, 'train.log')
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    OmegaConf.save(config, os.path.join(args.output_dir, 'config.yaml'))
    
    logger.info("Starting drug discovery model training")
    logger.info(f"Config: {config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    input_dim = train_loader.dataset[0][0].shape[0]
    model = create_model(
        model_type=config.model.type,
        input_dim=input_dim,
        **config.model.params
    )
    
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, **config.optimizer)
    scheduler = create_scheduler(optimizer, **config.scheduler) if config.get('scheduler') else None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=config.training.loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=get_device(),
        save_dir=args.output_dir,
        patience=config.training.patience,
        min_delta=config.training.min_delta
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train model
    training_history = trainer.train(config.training.num_epochs)
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=get_device(),
        save_dir=os.path.join(args.output_dir, 'evaluation')
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Test R²: {test_results['r2']:.4f}")
    logger.info(f"Test RMSE: {test_results['rmse']:.4f}")


if __name__ == "__main__":
    main()
