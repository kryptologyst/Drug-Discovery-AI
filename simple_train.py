"""
Simple training script for drug discovery models.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.core import set_seed, get_device
from src.data.molecular import generate_synthetic_dataset, smiles_to_morgan_fingerprint
from src.models import create_model


def main():
    """Simple training example."""
    print("Drug Discovery AI - Simple Training Example")
    print("=" * 50)
    
    # Set random seed
    set_seed(42)
    
    # Generate synthetic data
    print("Generating synthetic molecular dataset...")
    smiles_list, targets = generate_synthetic_dataset(n_samples=500, seed=42)
    
    # Create molecular features
    print("Creating molecular fingerprints...")
    features = []
    valid_targets = []
    
    for smiles, target in zip(smiles_list, targets):
        fp = smiles_to_morgan_fingerprint(smiles)
        if fp is not None:
            features.append(fp)
            valid_targets.append(target)
    
    features = np.array(features)
    targets = np.array(valid_targets)
    
    print(f"Created {len(features)} valid molecular samples")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest baseline
    print("\nTraining Random Forest baseline...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    rf_pred = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Random Forest R²: {rf_r2:.4f}")
    
    # Train Neural Network
    print("\nTraining Neural Network...")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create model
    model = create_model(
        model_type="fingerprint",
        input_dim=features.shape[1],
        hidden_dims=[128, 64],
        dropout=0.2
    )
    
    # Setup training
    device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        
        X_batch = X_train_tensor.to(device)
        y_batch = y_train_tensor.to(device)
        
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
    
    # Evaluate Neural Network
    print("\nEvaluating Neural Network...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        nn_pred = model(X_test_tensor).cpu().numpy()
    
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
    nn_r2 = r2_score(y_test, nn_pred)
    
    print(f"Neural Network RMSE: {nn_rmse:.4f}")
    print(f"Neural Network R²: {nn_r2:.4f}")
    
    # Compare models
    print("\nModel Comparison:")
    print("-" * 30)
    print(f"{'Model':<20} {'RMSE':<10} {'R²':<10}")
    print("-" * 30)
    print(f"{'Random Forest':<20} {rf_rmse:<10.4f} {rf_r2:<10.4f}")
    print(f"{'Neural Network':<20} {nn_rmse:<10.4f} {nn_r2:<10.4f}")
    print("-" * 30)
    
    # Test prediction on new molecule
    print("\nTesting prediction on new molecule...")
    test_smiles = "CCO"  # ethanol
    test_fp = smiles_to_morgan_fingerprint(test_smiles)
    
    if test_fp is not None:
        # Random Forest prediction
        rf_pred_new = rf_model.predict([test_fp])[0]
        
        # Neural Network prediction
        model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(test_fp, dtype=torch.float32).unsqueeze(0).to(device)
            nn_pred_new = model(test_tensor).cpu().item()
        
        print(f"Molecule: {test_smiles}")
        print(f"Random Forest prediction: {rf_pred_new:.4f}")
        print(f"Neural Network prediction: {nn_pred_new:.4f}")
    
    print("\nTraining completed successfully!")
    print("This is a research demonstration - not for clinical use.")


if __name__ == "__main__":
    main()
