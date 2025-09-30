#!/usr/bin/env python3
"""
🧠 TRAINING DATA GENERATION SCRIPT
==================================

Quick script to generate training data for robot navigation neural network.
This script creates a dataset following the specifications in data_generation_pipeline.md

Usage:
    python scripts/generate_data.py small
    python scripts/generate_data.py medium
    python scripts/generate_data.py large
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path so we can import from core
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation import TrainingDataGenerator, TrainingConfig, save_training_data


def generate_small_dataset(use_enhanced=True):
    """Generate a small dataset for quick testing"""
    print("🧠 Generating small training dataset...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=100,  # Small dataset for quick generation
        obstacle_density_range=(0.1, 0.3),
        min_path_length=5,
        max_path_length=25,
        history_length=3  # Solution 1: Memory/History
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset(use_enhanced=use_enhanced)
    
    # Save the data
    save_training_data(X_train, y_train, metadata, "data/raw/small_training_dataset.npz")
    
    # Show summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total examples: {len(X_train)}")
    print(f"   Input shape: {X_train.shape}")
    print(f"   Output shape: {y_train.shape}")
    print(f"   Environments: {len(metadata)}")
    
    # Show action distribution
    if len(y_train) > 0:
        action_counts = np.bincount(y_train.astype(int))
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"\n   Action distribution:")
        for count, name in zip(action_counts, action_names):
            percentage = (count / len(y_train)) * 100
            print(f"     {name}: {count} ({percentage:.1f}%)")
    else:
        print(f"\n   No training data generated")
    
    return X_train, y_train, metadata


def generate_medium_dataset(use_enhanced=True):
    """Generate a medium-sized dataset for training"""
    print("🧠 Generating medium training dataset...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=500,  # Medium dataset
        obstacle_density_range=(0.1, 0.35),
        min_path_length=5,
        max_path_length=30,
        history_length=3  # Solution 1: Memory/History
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset(use_enhanced=use_enhanced)
    
    # Save the data
    save_training_data(X_train, y_train, metadata, "data/raw/medium_training_dataset.npz")
    
    # Show summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total examples: {len(X_train)}")
    print(f"   Input shape: {X_train.shape}")
    print(f"   Output shape: {y_train.shape}")
    print(f"   Environments: {len(metadata)}")
    
    return X_train, y_train, metadata


def generate_large_dataset(use_enhanced=True):
    """Generate a large dataset for full training"""
    print("🧠 Generating large training dataset...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=1000,  # Large dataset
        obstacle_density_range=(0.1, 0.4),
        min_path_length=5,
        max_path_length=50,
        history_length=3  # Solution 1: Memory/History
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset(use_enhanced=use_enhanced)
    
    # Save the data
    save_training_data(X_train, y_train, metadata, "data/raw/large_training_dataset.npz")
    
    # Show summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total examples: {len(X_train)}")
    print(f"   Input shape: {X_train.shape}")
    print(f"   Output shape: {y_train.shape}")
    print(f"   Environments: {len(metadata)}")
    
    return X_train, y_train, metadata


def show_sample_data(X_train, y_train):
    """Show sample training data"""
    print(f"\n👁️ Sample Training Data:")
    print("=" * 50)
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    feature_size = X_train.shape[1] if len(X_train.shape) > 1 else 1
    
    # Show first 5 examples
    for i in range(min(5, len(X_train))):
        action = y_train[i]
        
        print(f"\nExample {i+1}:")
        
        if feature_size == 21:
            # Enhanced mode: 9 perception + 12 history
            perception = X_train[i][:9].reshape(3, 3)
            history = X_train[i][9:21].reshape(3, 4)  # 3 actions × 4 one-hot
            
            print(f"  3x3 Perception:")
            for row in perception:
                print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
            
            print(f"  Action History (last 3 actions):")
            for j, action_vec in enumerate(history):
                action_idx = np.argmax(action_vec) if np.sum(action_vec) > 0 else -1
                if action_idx >= 0:
                    print(f"    {j+1}. {action_names[action_idx]}")
                else:
                    print(f"    {j+1}. (no action)")
        else:
            # Basic mode: 9 perception only
            perception = X_train[i].reshape(3, 3)
            print(f"  3x3 Perception (flattened: {X_train[i]}):")
            for row in perception:
                print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
        
        print(f"  → Action: {action} ({action_names[action]})")


def main():
    """Main function to generate training data"""
    parser = argparse.ArgumentParser(description='Generate training data for robot navigation')
    parser.add_argument('dataset_size', choices=['small', 'medium', 'large'], 
                       help='Size of dataset to generate')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Directory to save the dataset')
    parser.add_argument('--basic', action='store_true', 
                       help='Use basic mode (9 features) instead of enhanced mode (21 features)')
    
    args = parser.parse_args()
    
    use_enhanced = not args.basic
    mode_str = "basic (9 features)" if args.basic else "enhanced (21 features)"
    
    print(f"🚀 Generating {args.dataset_size} training dataset ({mode_str})...")
    print("=" * 60)
    
    if args.dataset_size == "small":
        X_train, y_train, metadata = generate_small_dataset(use_enhanced=use_enhanced)
    elif args.dataset_size == "medium":
        X_train, y_train, metadata = generate_medium_dataset(use_enhanced=use_enhanced)
    elif args.dataset_size == "large":
        X_train, y_train, metadata = generate_large_dataset(use_enhanced=use_enhanced)
    else:
        print(f"❌ Unknown dataset size: {args.dataset_size}")
        print("Available options: small, medium, large")
        sys.exit(1)
    
    # Show sample data
    show_sample_data(X_train, y_train)
    
    print(f"\n✅ {args.dataset_size.title()} dataset generation complete!")
    print(f"   Dataset saved and ready for neural network training")
    print(f"   Use load_training_data() to load the dataset")


if __name__ == "__main__":
    main()
