"""
🤖 ROBOT NAVIGATION NEURAL NETWORK TRAINING
==========================================

This script demonstrates how to train the neural network for robot navigation
using the generated training data.

Usage:
    python scripts/train_nn.py

Features:
- Loads training data from data/raw/small_training_dataset.npz
- Creates train/validation/test splits
- Trains neural network with early stopping
- Saves trained model and training history
- Generates performance analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.pytorch_network import RobotNavigationNet, RobotNavigationTrainer, load_config, create_data_loaders
from core.data_generation import load_training_data

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

<<<<<<< Updated upstream
def load_config(config_path: str = None) -> dict:
=======
def get_dataset_filename(use_goal_delta: bool = True) -> str:
>>>>>>> Stashed changes
    """
    Load configuration from YAML file
    
    Args:
<<<<<<< Updated upstream
        config_path: Path to configuration file
        
=======
        use_goal_delta: True for goal-aware mode, False for basic mode
    
>>>>>>> Stashed changes
    Returns:
        Configuration dictionary
    """
<<<<<<< Updated upstream
    if config_path is None:
        config_path = project_root / "configs" / "nn_config.yaml"
    
=======
    if use_goal_delta:
        return "large_training_dataset.npz"  # Goal-aware mode (default)
    else:
        return "large_training_dataset_basic.npz"  # Basic mode

def load_data(use_goal_delta: bool = True, verbose: bool = True):
    """
    Load training data with automatic mode detection
    
    Args:
        use_goal_delta: True for goal-aware mode, False for basic mode
        verbose: Print loading information
    
    Returns:
        X, y, metadata, is_goal_aware
    """
    # Get filename
    data_filename = get_dataset_filename(use_goal_delta)
    data_path = project_root / "data" / "raw" / data_filename
    
    if verbose:
        print(f"📂 Loading data: {data_filename}")
    
    # Load data
>>>>>>> Stashed changes
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
<<<<<<< Updated upstream
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Using default configuration...")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML configuration: {e}")
        print("💡 Using default configuration...")
        return get_default_config()
=======
        print(f"❌ Data file not found: {data_filename}")
        print(f"💡 Generate data first with:")
        if use_goal_delta:
            print(f"   python scripts/generate_data.py large")
        else:
            print(f"   python scripts/generate_data.py large --basic")
        raise
    
    # Detect mode from data
    feature_count = X.shape[1]
    is_goal_aware = feature_count == 11
    
    if verbose:
        mode_type = "Goal-Aware 🎯" if is_goal_aware else "Basic"
        print(f"✅ Data loaded: {X.shape[0]} samples")
        print(f"   Features: {X.shape[1]} ({mode_type})")
        print(f"   Environments: {len(metadata)}")
    
    return X, y, metadata, is_goal_aware
>>>>>>> Stashed changes

def get_default_config() -> dict:
    """Get default configuration if YAML file is not available"""
    return {
        'model': {
            'input_size': 9,
            'hidden1_size': 64,
            'hidden2_size': 32,
            'output_size': 4,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'early_stopping_patience': 15
        },
        'data': {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        },
        'model_save': {
            'model_dir': 'data/models/final_models',
            'model_filename': 'robot_navigation_nn.pkl',
            'history_filename': 'training_history.png'
        }
    }

# =============================================================================
# FILE PATHS
# =============================================================================

# Default file paths
DATA_PATH = project_root / "data" / "raw" / "small_training_dataset.npz"

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

<<<<<<< Updated upstream
def main(config_path: str = None, perception_mode: str = "3x3"):
=======
def train_model(use_goal_delta: bool = True,
                config_path: str = None,
                verbose: bool = True):
>>>>>>> Stashed changes
    """
    Main training pipeline
    
    Args:
<<<<<<< Updated upstream
        config_path: Path to configuration file
        perception_mode: "3x3" or "5x5" perception mode
    """
    print("🤖 ROBOT NAVIGATION NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # 1. Load configuration
    print("📋 Loading configuration...")
    config = load_config(config_path, perception_mode)
    
    # Extract configuration sections
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    save_config = config['model_save']
    
    # 2. Load training data
    print("\n📂 Loading training data...")
    try:
        X, y = load_training_data(DATA_PATH)
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Data types: X={X.dtype}, y={y.dtype}")
        print(f"🎯 Action distribution: {np.bincount(y)}")
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_PATH}")
        print("💡 Run 'python scripts/generate_data.py' first to generate training data")
        return
=======
        use_goal_delta: True for goal-aware mode, False for basic mode
        config_path: Path to config file (None = default)
        verbose: Print training progress
    
    Returns:
        trainer, history, test_metrics
    """
    if verbose:
        print("🚀 ROBOT NAVIGATION NEURAL NETWORK TRAINING")
        print("=" * 60)
        print(f"   Mode: {'Goal-Aware' if use_goal_delta else 'Basic'}")
        print(f"   Features: {'11 (9 perception + 2 goal_delta)' if use_goal_delta else '9 (perception only)'}")
    
    # 1. Load configuration
    config = load_config(goal_aware=use_goal_delta)
    
    # 2. Load data
    X, y, metadata, is_goal_aware = load_data(use_goal_delta, verbose)
>>>>>>> Stashed changes
    
    # 3. Create data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y,
        batch_size=training_config['batch_size'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio']
    )
    
    # 4. Initialize neural network
    print("\n🧠 Initializing PyTorch neural network...")
    model = RobotNavigationNet(
        input_size=model_config['input_size'],
        hidden1_size=model_config['hidden1_size'],
        hidden2_size=model_config['hidden2_size'],
        output_size=model_config['output_size'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Create trainer
    trainer = RobotNavigationTrainer(
        model=model,
        learning_rate=model_config['learning_rate']
    )
    
    print(f"✅ Model created with architecture:")
    print(f"   {model.get_architecture_info()['architecture']}")
    print(f"   Total parameters: {model.get_architecture_info()['total_parameters']}")
    print(f"   Device: {trainer.device}")
    
    # 5. Train the model
    print("\n🚀 Starting training...")
    history = trainer.train(
        train_loader, val_loader,
        epochs=training_config['epochs'],
        early_stopping_patience=training_config['early_stopping_patience']
    )
    
    # 6. Evaluate on test set
    print("\n🎯 Evaluating on test set...")
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    print(f"📊 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy:.2f}%")
    
    # 7. Detailed analysis
    print("\n📈 Detailed Performance Analysis:")
    # Get predictions for analysis
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(trainer.device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    print(f"   Manual accuracy calculation: {accuracy:.4f}")
    
    # Action distribution
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    print(f"   Action distribution in test set: {np.bincount(all_targets)}")
    print(f"   Predicted action distribution: {np.bincount(all_predictions)}")
    
    # 8. Save model and results
    print("\n💾 Saving model and results...")
    
    # Create file paths from config
    model_dir = project_root / save_config['model_dir']
    model_path = model_dir / save_config['model_filename']
    history_path = project_root / "data" / "results" / "visualizations" / save_config['history_filename']
    
    # Create directories if they don't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    trainer.save_model(str(model_path))
    
    # Save training history plot
    trainer.plot_training_history(str(history_path))
    
    # 9. Print summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"💾 Model saved to: {model_path}")
    print(f"📈 Training history saved to: {history_path}")
    print(f"🧠 Architecture: {model.get_architecture_info()['architecture']}")
    print(f"⚙️  Hyperparameters: lr={model_config['learning_rate']}, dropout={model_config['dropout_rate']}")
    print(f"🖥️  Device: {trainer.device}")
    
    # Training statistics
    final_epoch = len(history['train_losses'])
    best_val_loss = min(history['val_losses'])
    best_val_acc = max(history['val_accuracies'])
    
    print(f"\n📈 Training Statistics:")
    print(f"   Epochs trained: {final_epoch}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    
    return trainer, history

# =============================================================================
# HYPERPARAMETER TUNING EXAMPLE
# =============================================================================

def hyperparameter_tuning_example():
    """
    Example of how to perform hyperparameter tuning
    """
    print("\n🔧 HYPERPARAMETER TUNING EXAMPLE")
    print("=" * 40)
    
    # Load data
    X, y = load_training_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
    
    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.1, 0.2, 0.3]
    hidden_sizes = [(32, 16), (64, 32), (128, 64)]
    
    best_accuracy = 0
    best_config = None
    
    print("🔍 Testing hyperparameter combinations...")
    
    for lr in learning_rates:
        for dropout in dropout_rates:
            for h1_size, h2_size in hidden_sizes:
                print(f"\n🧪 Testing: lr={lr}, dropout={dropout}, hidden=({h1_size}, {h2_size})")
                
                # Create model with current hyperparameters
                model = RobotNavigationNN(
                    input_size=9,
                    hidden1_size=h1_size,
                    hidden2_size=h2_size,
                    output_size=4,
                    dropout_rate=dropout,
                    learning_rate=lr
                )
                
                # Train for fewer epochs for tuning
                model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
                
                # Evaluate
                _, val_accuracy = model.evaluate(X_val, y_val)
                print(f"   Validation accuracy: {val_accuracy:.4f}")
                
                # Track best configuration
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_config = {
                        'learning_rate': lr,
                        'dropout_rate': dropout,
                        'hidden1_size': h1_size,
                        'hidden2_size': h2_size
                    }
    
    print(f"\n🏆 Best configuration:")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   Config: {best_config}")
    
    return best_config, best_accuracy

# =============================================================================
# MODEL COMPARISON EXAMPLE
# =============================================================================

def compare_architectures():
    """
    Compare different neural network architectures
    """
    print("\n🏗️ ARCHITECTURE COMPARISON")
    print("=" * 40)
    
    # Load data
    X, y = load_training_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
    
    # Different architectures to test
    architectures = [
        {"name": "Small", "hidden1": 32, "hidden2": 16},
        {"name": "Medium", "hidden1": 64, "hidden2": 32},
        {"name": "Large", "hidden1": 128, "hidden2": 64},
        {"name": "Deep", "hidden1": 64, "hidden2": 32, "hidden3": 16}
    ]
    
    results = []
    
    for arch in architectures:
        print(f"\n🧪 Testing {arch['name']} architecture...")
        
        # Create model (simplified for this example)
        model = RobotNavigationNN(
            input_size=9,
            hidden1_size=arch['hidden1'],
            hidden2_size=arch['hidden2'],
            output_size=4,
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Train briefly
        model.train(X_train, y_train, X_val, y_val, epochs=20)
        
        # Evaluate
        _, test_accuracy = model.evaluate(X_test, y_test)
        results.append({
            'name': arch['name'],
            'accuracy': test_accuracy,
            'params': arch['hidden1'] * arch['hidden2']  # Approximate parameter count
        })
        
        print(f"   Test accuracy: {test_accuracy:.4f}")
    
    # Print comparison
    print(f"\n📊 Architecture Comparison:")
    print("Architecture | Accuracy | Parameters")
    print("-" * 40)
    for result in results:
        print(f"{result['name']:12} | {result['accuracy']:.4f}   | {result['params']:10}")
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run the training pipeline with command line arguments
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Robot Navigation Neural Network')
<<<<<<< Updated upstream
=======
    parser.add_argument('--basic', action='store_true',
                       help='Use basic mode (9 features) instead of goal-aware mode (11 features)')
>>>>>>> Stashed changes
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: configs/nn_config.yaml)')
    parser.add_argument('--perception', choices=['3x3', '5x5'], default='3x3',
                       help='Perception window size (auto-selects config)')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Run hyperparameter tuning experiments')
    parser.add_argument('--architecture-comparison', action='store_true',
                       help='Run architecture comparison experiments')
    
    args = parser.parse_args()
    
    # Use unified config with perception mode
    if args.config is None:
        args.config = 'configs/nn_config.yaml'
        print(f"🎯 Using unified config with {args.perception} perception mode: {args.config}")
    
    try:
<<<<<<< Updated upstream
        # Main training with perception mode
        model, history = main(args.config, perception_mode=args.perception)
=======
        # Train model
        trainer, history, metrics = train_model(
            use_goal_delta=not args.basic,
            config_path=args.config,
            verbose=True
        )
>>>>>>> Stashed changes
        
        # Optional: Run additional experiments
        if args.hyperparameter_tuning or args.architecture_comparison:
            print("\n" + "=" * 60)
            print("🔬 ADDITIONAL EXPERIMENTS")
            print("=" * 60)
        
<<<<<<< Updated upstream
        if args.hyperparameter_tuning:
            print("\n🔧 Running hyperparameter tuning...")
            best_config, best_acc = hyperparameter_tuning_example()
=======
        mode_type = "basic" if args.basic else "goal_aware"
        model_path = model_dir / f"robot_nav_{mode_type}.pth"
>>>>>>> Stashed changes
        
        if args.architecture_comparison:
            print("\n🏗️ Running architecture comparison...")
            arch_results = compare_architectures()
        
<<<<<<< Updated upstream
        print("\n✅ All experiments completed!")
=======
        # Save training history plot
        vis_dir = project_root / "data" / "results" / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        history_path = vis_dir / f"training_{mode_type}.png"
        
        trainer.plot_training_history(str(history_path))
        print(f"📈 Training history saved to: {history_path}")
        
        print("\n🎉 Training completed successfully!")
>>>>>>> Stashed changes
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
