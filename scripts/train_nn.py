#!/usr/bin/env python3
"""
🤖 Robot Navigation Neural Network Training Script
==================================================

Train neural networks for goal-aware robot navigation.

Usage:
    python scripts/train_nn.py                    # Goal-aware mode (11 features)
    python scripts/train_nn.py --basic            # Basic mode (9 features)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation import load_training_data
from core.pytorch_network import load_config, RobotNavigationNet


def get_dataset_filename(use_goal_delta: bool = True) -> str:
    """Get the appropriate dataset filename based on mode"""
    if use_goal_delta:
        return "large_training_dataset.npz"
    else:
        return "large_training_dataset_basic.npz"


def load_data(use_goal_delta: bool = True, verbose: bool = True):
    """
    Load training data for robot navigation
    
    Args:
        use_goal_delta: If True, use goal-aware mode (11 features), else basic mode (9 features)
        verbose: Whether to print loading information
        
    Returns:
        X: Input features
        y: Target actions
        metadata: Environment metadata
        is_goal_aware: Boolean indicating if goal-aware mode was detected
    """
    # Determine data filename
    data_filename = get_dataset_filename(use_goal_delta)
    data_path = project_root / "data" / "raw" / data_filename
    
    if verbose:
        print(f"📂 Loading data: {data_filename}")
    
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print(f"💡 Generate data first: python scripts/generate_data.py large{' --basic' if not use_goal_delta else ''}")
        sys.exit(1)
    
    # Load data
    X, y, metadata = load_training_data(str(data_path))
    
    # Detect actual mode from data
    feature_count = X.shape[1]
    is_goal_aware = feature_count == 11
    
    if verbose:
        mode_type = "Goal-Aware 🎯" if is_goal_aware else "Basic"
        print(f"✅ Data loaded: {len(X)} samples")
        print(f"   Features: {feature_count} ({mode_type})")
        print(f"   Environments: {len(metadata)}")
    
    return X, y, metadata, is_goal_aware


def prepare_data_loaders(X, y, config, test_size=0.1, val_size=0.1, random_state=42):
    """
    Prepare data loaders for training, validation, and testing
    
    Args:
        X: Input features
        y: Target actions
        config: Model configuration
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if len(X) > 0:  # Only print if we have data
        print(f"📊 Data splits:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
    
    return train_loader, val_loader, test_loader


def create_model(config):
    """Create neural network model"""
    model = RobotNavigationNet(
        input_size=config['model']['input_size'],
        hidden1_size=config['model']['hidden1_size'],
        hidden2_size=config['model']['hidden2_size'],
        output_size=config['model']['output_size'],
        dropout_rate=config['model']['dropout_rate']
    )
    return model


def create_trainer(model, config):
    """Create trainer with optimizer and loss function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    class Trainer:
        def __init__(self, model, optimizer, criterion, device):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
        
        def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15, verbose=True):
            """Train the model"""
            history = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}
            best_val_loss = float('inf')
            patience_counter = 0
            
            if verbose:
                print(f"🚀 Starting training on {device}")
                print(f"🧠 Architecture: {self.model.get_architecture_info()['architecture']}")
                print(f"⚙️  Learning rate: {self.optimizer.param_groups[0]['lr']}")
                print(f"🛡️  Dropout rate: {self.model.dropout_rate}")
                print("-" * 60)
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
                
                # Validation phase
                val_loss, val_accuracy = self.evaluate(val_loader)
                
                # Record history
                train_accuracy = train_correct / train_total
                history['train_losses'].append(train_loss / len(train_loader))
                history['train_accuracies'].append(train_accuracy)
                history['val_losses'].append(val_loss)
                history['val_accuracies'].append(val_accuracy)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: Train Loss={history['train_losses'][-1]:.4f}, Train Acc={train_accuracy:.2%}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2%}")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"🛑 Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                        print(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")
                    break
            
            return history
        
        def evaluate(self, data_loader):
            """Evaluate model on data"""
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in data_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            return total_loss / len(data_loader), correct / total
    
    return Trainer(model, optimizer, criterion, device)


def get_predictions(model, data_loader, device):
    """Get model predictions"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)


def calculate_metrics(predictions, targets):
    """Calculate classification metrics"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')
    f1 = f1_score(targets, predictions, average='macro')
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def train_model(use_goal_delta: bool = True, verbose: bool = True):
    """Main training function"""
    # Load configuration
    config = load_config(goal_aware=use_goal_delta)
    
    # Load data
    X, y, metadata, is_goal_aware = load_data(use_goal_delta, verbose)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(X, y, config)
    
    # Create model and trainer
    model = create_model(config)
    trainer = create_trainer(model, config)
    
    if verbose:
        mode_type = "Goal-Aware" if is_goal_aware else "Basic"
        print(f"🧠 Training {mode_type} Neural Network")
        print(f"📊 Input features: {X.shape[1]}")
    
    # Train model
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=config['training']['epochs'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        verbose=verbose
    )
    
    # Evaluate on test set
    test_accuracy, test_loss = trainer.evaluate(test_loader)
    
    if verbose:
        print(f"\n📊 Test Set Performance:")
        print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"   Loss: {test_loss:.4f}")
    
    # Calculate additional metrics
    predictions, targets = get_predictions(model, test_loader, trainer.device)
    metrics = calculate_metrics(predictions, targets)
    
    return trainer, history, metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Robot Navigation Neural Network')
    parser.add_argument('--basic', action='store_true',
                       help='Use basic mode (9 features) instead of goal-aware mode (11 features)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Determine mode
    use_goal_delta = not args.basic
    mode_type = "Goal-Aware" if use_goal_delta else "Basic"
    
    print(f"🤖 ROBOT NAVIGATION NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"   Mode: {mode_type}")
    print(f"   Features: {'11 (9 perception + 2 goal_delta)' if use_goal_delta else '9 (perception only)'}")
    print(f"   Expected Accuracy: {'80-85%' if use_goal_delta else '70-75%'}")
    print()
    
    try:
        # Train model
        trainer, history, metrics = train_model(
            use_goal_delta=use_goal_delta,
            verbose=True
        )
        
        print(f"\n📋 Detailed Metrics:")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        
        # Save model
        model_dir = project_root / "data" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        mode_type = "basic" if args.basic else "goal_aware"
        model_path = model_dir / f"robot_navigation_{mode_type}.pth"
        
        torch.save(trainer.model.state_dict(), model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save training history plot
        vis_dir = project_root / "data" / "results" / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = vis_dir / f"training_history_{mode_type}.png"
        
        # Create and save plot
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_losses'], label='Training Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax1.set_title(f'Loss - {mode_type.title()} Mode')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_accuracies'], label='Training Accuracy')
        ax2.plot(history['val_accuracies'], label='Validation Accuracy')
        ax2.set_title(f'Accuracy - {mode_type.title()} Mode')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history plot saved to: {history_path}")
        print(f"\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()