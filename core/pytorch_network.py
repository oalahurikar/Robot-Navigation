"""
🧠 ROBOT NAVIGATION NEURAL NETWORK - PYTORCH IMPLEMENTATION
===========================================================

Biological Inspiration:
- Mimics animal navigation with limited 3×3 perception
- ReLU activation mimics spiking neurons
- Dropout simulates neural noise and robustness

Mathematical Foundation:
- Softmax + Cross-entropy for multi-class classification
- ReLU for gradient flow and sparsity
- Dropout for regularization

Learning Objective:
Train robot to map 3×3 obstacle patterns to navigation actions using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import yaml
from pathlib import Path
import pickle
import os

# =============================================================================
# BIOLOGICAL BACKGROUND
# =============================================================================
"""
🧠 NEUROSCIENCE CONNECTION:
- Local perception: Animals use limited peripheral vision for navigation
- Pattern recognition: Brain learns obstacle-action relationships
- Robustness: Neural networks work despite individual neuron failures
- Competition: Softmax mimics neural competition for activation
"""

# =============================================================================
# MATHEMATICAL FOUNDATION
# =============================================================================
"""
📐 KEY EQUATIONS:
- ReLU: f(x) = max(0, x) - Simple, biologically plausible
- Softmax: p_i = exp(z_i) / Σ exp(z_j) - Probability normalization
- Cross-entropy: L = -Σ y_i log(p_i) - Classification loss
- Dropout: x' = x * mask / (1-p) - Regularization technique
"""

# =============================================================================
# PYTORCH NEURAL NETWORK IMPLEMENTATION
# =============================================================================

class RobotNavigationNet(nn.Module):
    """
    PyTorch Neural Network for Robot Navigation
    
    Architecture:
    Input(9) → Hidden1(64) → Hidden2(32) → Output(4)
    ReLU + Dropout(0.2) + Softmax
    """
    
    def __init__(self, 
                 input_size: int = 9,
                 hidden1_size: int = 64,
                 hidden2_size: int = 32,
                 output_size: int = 4,
                 dropout_rate: float = 0.2):
        """
        Initialize PyTorch neural network
        
        Args:
            input_size: 3×3 perception grid (9 neurons)
            hidden1_size: First hidden layer (64 neurons)
            hidden2_size: Second hidden layer (32 neurons) 
            output_size: 4 navigation actions
            dropout_rate: Regularization strength (0.2)
        """
        super(RobotNavigationNet, self).__init__()
        
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization for stable gradients"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor (batch_size, 9)
            
        Returns:
            output: Predicted probabilities (batch_size, 4)
        """
        # Input layer (no activation needed)
        x = x.view(x.size(0), -1)  # Flatten if needed
        
        # Hidden layer 1: Linear → ReLU → Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Hidden layer 2: Linear → ReLU → Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer: Linear → Softmax
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def predict(self, x):
        """Make predictions (inference mode)"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            output = self.forward(x)
            predictions = torch.argmax(output, dim=1)
        return predictions
    
    def get_architecture_info(self):
        """Get information about the network architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': f"{self.input_size} → {self.hidden1_size} → {self.hidden2_size} → {self.output_size}",
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate
        }


class RobotNavigationTrainer:
    """
    PyTorch trainer for robot navigation neural network
    """
    
    def __init__(self, 
                 model: RobotNavigationNet,
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch neural network
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs: int = 100,
              early_stopping_patience: int = 15,
              verbose: bool = True):
        """
        Train the neural network
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            verbose: Print training progress
            
        Returns:
            history: Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            print(f"🚀 Starting training on {self.device}")
            print(f"🧠 Architecture: {self.model.get_architecture_info()['architecture']}")
            print(f"⚙️  Learning rate: {self.learning_rate}")
            print(f"🛡️  Dropout rate: {self.model.dropout_rate}")
            print("-" * 60)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"🛑 Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                break
        
        if verbose:
            print(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        test_loss, test_acc = self.validate(test_loader)
        return test_loss, test_acc
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'architecture_info': self.model.get_architecture_info(),
            'history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            }
        }
        
        torch.save(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(model_data['model_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        # Load history if available
        if 'history' in model_data:
            self.train_losses = model_data['history']['train_losses']
            self.val_losses = model_data['history']['val_losses']
            self.train_accuracies = model_data['history']['train_accuracies']
            self.val_accuracies = model_data['history']['val_accuracies']
        
        print(f"📂 Model loaded from {filepath}")


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "nn_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Using default configuration...")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML configuration: {e}")
        print("💡 Using default configuration...")
        return get_default_config()

def get_default_config() -> dict:
    """Get default configuration if YAML file is not available"""
    return {
        'model': {
            'input_size': 9,
            'hidden1_size': 64,
            'hidden2_size': 32,
            'output_size': 4,
            'dropout_rate': 0.2
        },
        'training': {
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'early_stopping_patience': 15
        },
        'data': {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        }
    }


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def create_data_loaders(X: np.ndarray, 
                       y: np.ndarray,
                       batch_size: int = 32,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1,
                       random_seed: int = 42):
    """
    Create PyTorch data loaders from numpy arrays
    
    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        batch_size: Batch size for data loaders
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_tensor, y_tensor, 
        test_size=test_ratio, 
        random_state=random_seed,
        stratify=y_tensor
    )
    
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_seed,
        stratify=y_temp
    )
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Data splits:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the PyTorch neural network
    """
    print("🤖 Robot Navigation Neural Network - PyTorch Implementation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    model_config = config['model']
    training_config = config['training']
    
    # Create model
    model = RobotNavigationNet(**model_config)
    print(f"✅ Model created: {model.get_architecture_info()['architecture']}")
    print(f"📊 Total parameters: {model.get_architecture_info()['total_parameters']}")
    
    # Create trainer
    trainer = RobotNavigationTrainer(
        model=model,
        learning_rate=training_config['learning_rate']
    )
    print(f"✅ Trainer created with learning rate: {trainer.learning_rate}")
    print(f"🖥️  Device: {trainer.device}")
