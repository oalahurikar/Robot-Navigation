"""
🤖 ROBOT NAVIGATION NEURAL NETWORK TRAINING
==========================================

This module provides all the core functionality for training the robot navigation
neural network. It handles:
- Data loading and preprocessing
- Model creation and configuration
- Training pipeline with early stopping
- Model evaluation and metrics
- Model saving and loading

Import this module in notebooks for interactive training and visualization.

Usage in script:
    python scripts/train_nn.py --perception 5x5 --distance

Usage in notebook:
    from scripts.train_nn import *
    trainer, history = train_model(perception_mode='5x5', use_distance=True)
"""

import sys
import numpy as np
import torch
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.pytorch_network import RobotNavigationNet, RobotNavigationTrainer, load_config, create_data_loaders
from core.data_generation import load_training_data

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def get_dataset_filename(perception_mode: str = '5x5', use_distance: bool = True) -> str:
    """
    Determine the correct dataset filename based on configuration
    
    Args:
        perception_mode: '3x3' or '5x5'
        use_distance: True for distance-based, False for binary
    
    Returns:
        Dataset filename
    """
    if use_distance:
        return f"distance_{perception_mode}_large.npz"
    else:
        return f"binary_{perception_mode}_large.npz" if perception_mode == '5x5' else "large_training_dataset.npz"

def load_data(perception_mode: str = '5x5', use_distance: bool = True, verbose: bool = True):
    """
    Load training data with automatic perception type detection
    
    Args:
        perception_mode: '3x3' or '5x5'
        use_distance: True for distance-based, False for binary
        verbose: Print loading information
    
    Returns:
        X, y, metadata, is_distance_based
    """
    # Get filename
    data_filename = get_dataset_filename(perception_mode, use_distance)
    data_path = project_root / "data" / "raw" / data_filename
    
    if verbose:
        print(f"📂 Loading data: {data_filename}")
    
    # Load data
    try:
        X, y, metadata = load_training_data(data_path)
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_filename}")
        print(f"💡 Generate data first with:")
        if use_distance:
            print(f"   python scripts/generate_data.py large --perception {perception_mode} --distance")
        else:
            print(f"   python scripts/generate_data.py large --perception {perception_mode}")
        raise
    
    # Detect perception type from data
    perception_size = 25 if perception_mode == '5x5' else 9
    sample_perception = X[0][:perception_size]
    unique_vals = np.unique(sample_perception)
    is_distance_based = len(unique_vals) > 2 or (len(unique_vals) == 2 and not (0.0 in unique_vals and 1.0 in unique_vals))
    
    if verbose:
        perception_type = "Distance-based 🎯" if is_distance_based else "Binary"
        print(f"✅ Data loaded: {X.shape[0]} samples")
        print(f"   Features: {X.shape[1]} ({perception_mode} {perception_type})")
        print(f"   Environments: {len(metadata)}")
    
    return X, y, metadata, is_distance_based

def prepare_data_loaders(X, y, config, verbose: bool = True):
    """
    Create PyTorch data loaders from numpy arrays
    
    Args:
        X: Input features
        y: Target labels
        config: Configuration dictionary
        verbose: Print split information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y,
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    if verbose:
        print(f"\n📊 Data Splits:")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL CREATION UTILITIES
# =============================================================================

def create_model(config, verbose: bool = True):
    """
    Create neural network model from configuration
    
    Args:
        config: Configuration dictionary
        verbose: Print model information
    
    Returns:
        model: RobotNavigationNet instance
    """
    model = RobotNavigationNet(
        input_size=config['model']['input_size'],
        hidden1_size=config['model']['hidden1_size'],
        hidden2_size=config['model']['hidden2_size'],
        output_size=config['model']['output_size'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    if verbose:
        info = model.get_architecture_info()
        print(f"\n🧠 Model Architecture:")
        print(f"   {info['architecture']}")
        print(f"   Total parameters: {info['total_parameters']:,}")
        print(f"   Dropout rate: {info['dropout_rate']}")
    
    return model

def create_trainer(model, config, verbose: bool = True):
    """
    Create trainer from model and configuration
    
    Args:
        model: Neural network model
        config: Configuration dictionary
        verbose: Print trainer information
    
    Returns:
        trainer: RobotNavigationTrainer instance
    """
    trainer = RobotNavigationTrainer(
        model=model,
        learning_rate=config['training']['learning_rate']
    )
    
    if verbose:
        print(f"\n⚙️  Trainer Configuration:")
        print(f"   Learning rate: {config['training']['learning_rate']}")
        print(f"   Device: {trainer.device}")
    
    return trainer

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_model(perception_mode: str = '5x5', 
                use_distance: bool = True,
                config_path: str = None,
                verbose: bool = True):
    """
    Complete training pipeline
    
    Args:
        perception_mode: '3x3' or '5x5'
        use_distance: True for distance-based, False for binary
        config_path: Path to config file (None = default)
        verbose: Print training progress
    
    Returns:
        trainer, history, test_metrics
    """
    if verbose:
        print("🚀 ROBOT NAVIGATION NEURAL NETWORK TRAINING")
        print("=" * 60)
        print(f"   Perception Mode: {perception_mode}")
        print(f"   Perception Type: {'Distance-based' if use_distance else 'Binary'}")
    
    # 1. Load configuration
    config = load_config(perception_mode=perception_mode)
    
    # 2. Load data
    X, y, metadata, is_distance = load_data(perception_mode, use_distance, verbose)
    
    # 3. Create data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(X, y, config, verbose)
    
    # 4. Create model
    model = create_model(config, verbose)
    
    # 5. Create trainer
    trainer = create_trainer(model, config, verbose)
    
    # 6. Train
    if verbose:
        print(f"\n🔥 Starting training...")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Early stopping patience: {config['training']['early_stopping']['patience']}")
    
    history = trainer.train(
        train_loader, 
        val_loader,
        epochs=config['training']['epochs'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        verbose=verbose
    )
    
    # 7. Evaluate on test set
    test_accuracy, test_loss = trainer.evaluate(test_loader)
    
    if verbose:
        print(f"\n✅ Training completed!")
        print(f"📊 Final Results:")
        print(f"   Best validation accuracy: {max(history['val_accuracies']):.4f}")
        print(f"   Test accuracy: {test_accuracy:.4f}")
        print(f"   Test loss: {test_loss:.4f}")
    
    test_metrics = {
        'accuracy': test_accuracy,
        'loss': test_loss
    }
    
    return trainer, history, test_metrics

# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def get_predictions(model, data_loader, device='cpu'):
    """
    Get model predictions on a dataset
    
    Args:
        model: Trained model
        data_loader: PyTorch DataLoader
        device: Device to use
    
    Returns:
        predictions, targets (as numpy arrays)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            outputs = model(data)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)

def calculate_metrics(predictions, targets):
    """
    Calculate detailed metrics from predictions
    
    Args:
        predictions: Predicted labels
        targets: True labels
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# =============================================================================
# MAIN SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Command-line interface for training
    """
    parser = argparse.ArgumentParser(description='Train Robot Navigation Neural Network')
    parser.add_argument('--perception', choices=['3x3', '5x5'], default='5x5',
                       help='Perception window size')
    parser.add_argument('--distance', action='store_true',
                       help='Use distance-based perception (default: binary)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Train model
        trainer, history, metrics = train_model(
            perception_mode=args.perception,
            use_distance=args.distance,
            config_path=args.config,
            verbose=True
        )
        
        # Save model
        model_dir = project_root / "data" / "models" / "final_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        perception_type = "distance" if args.distance else "binary"
        model_path = model_dir / f"robot_nav_{args.perception}_{perception_type}.pth"
        
        trainer.save_model(str(model_path))
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save training history plot
        vis_dir = project_root / "data" / "results" / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        history_path = vis_dir / f"training_{args.perception}_{perception_type}.png"
        
        trainer.plot_training_history(str(history_path))
        print(f"📈 Training history saved to: {history_path}")
        
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
