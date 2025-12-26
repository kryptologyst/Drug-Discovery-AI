"""
Evaluation utilities for drug discovery models.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from ..utils.core import get_device
from ..metrics import QSARMetrics, CalibrationMetrics, create_evaluation_plots, print_metrics_summary


class Evaluator:
    """Evaluator class for drug discovery models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device()
        self.save_dir = save_dir
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Setup metrics
        self.metrics = QSARMetrics()
        self.calibration_metrics = CalibrationMetrics()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create save directory
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def evaluate(self, return_predictions: bool = False) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        self.logger.info("Starting evaluation...")
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Evaluating")
            
            for batch in progress_bar:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
                elif torch.is_tensor(batch):
                    batch = batch.to(self.device)
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(batch, (list, tuple)):
                    predictions = self.model(*batch)
                    targets = batch[1] if len(batch) > 1 else batch[0]
                else:
                    predictions = self.model(batch)
                    targets = getattr(batch, 'y', None)
                
                if targets is not None:
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
                    # Handle uncertainty if model supports it
                    if hasattr(self.model, 'predict_with_uncertainty'):
                        _, lower, upper = self.model.predict_with_uncertainty(batch)
                        uncertainties = (upper - lower) / 2  # Half-width of CI
                        all_uncertainties.append(uncertainties.cpu().numpy())
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        uncertainties = np.concatenate(all_uncertainties) if all_uncertainties else None
        
        # Update metrics
        self.metrics.update(predictions, targets, uncertainties)
        
        # Compute metrics
        metrics = self.metrics.compute()
        
        # Add calibration metrics
        ece = self.calibration_metrics.compute_ece(predictions, targets)
        metrics['ece'] = ece
        
        # Create evaluation plots
        if self.save_dir:
            plots = create_evaluation_plots(
                predictions, targets, uncertainties, 
                save_path=self.save_dir
            )
            metrics['plots'] = plots
        
        # Print summary
        print_metrics_summary(metrics)
        
        # Save results
        if self.save_dir:
            self.save_results(metrics, predictions, targets, uncertainties)
        
        if return_predictions:
            return metrics, predictions, targets, uncertainties
        
        return metrics
    
    def save_results(
        self,
        metrics: Dict[str, Any],
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> None:
        """Save evaluation results."""
        # Save metrics
        metrics_path = os.path.join(self.save_dir, 'metrics.json')
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metrics_json[key] = float(value)
            elif key != 'plots':  # Skip plots for JSON
                metrics_json[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        # Save predictions
        results = {
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
        }
        
        if uncertainties is not None:
            results['uncertainties'] = uncertainties.tolist()
        
        results_path = os.path.join(self.save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {self.save_dir}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to evaluate a model."""
    evaluator = Evaluator(model, test_loader, device, save_dir)
    return evaluator.evaluate()


def compare_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple models."""
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        evaluator = Evaluator(model, test_loader, device, save_dir)
        results[name] = evaluator.evaluate()
    
    # Create comparison plot
    if save_dir:
        create_comparison_plot(results, save_dir)
    
    return results


def create_comparison_plot(
    results: Dict[str, Dict[str, Any]],
    save_dir: str
) -> None:
    """Create comparison plot for multiple models."""
    metrics_to_plot = ['rmse', 'mae', 'r2', 'pearson_r']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        model_names = list(results.keys())
        metric_values = [results[name].get(metric, 0) for name in model_names]
        
        bars = ax.bar(model_names, metric_values)
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(metric.upper())
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_leaderboard(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> 'pd.DataFrame':
    """Create a leaderboard from model comparison results."""
    import pandas as pd
    
    # Extract key metrics
    leaderboard_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        for metric in ['rmse', 'mae', 'r2', 'pearson_r', 'spearman_r']:
            row[metric.upper()] = metrics.get(metric, 0)
        leaderboard_data.append(row)
    
    leaderboard = pd.DataFrame(leaderboard_data)
    
    # Sort by R² score (descending)
    leaderboard = leaderboard.sort_values('R2', ascending=False)
    
    # Save if path provided
    if save_path:
        leaderboard.to_csv(save_path, index=False)
    
    return leaderboard
