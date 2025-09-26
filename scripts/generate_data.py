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


def generate_small_dataset():
    """Generate a small dataset for quick testing"""
    print("🧠 Generating small training dataset...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=100,  # Small dataset for quick generation
        obstacle_density_range=(0.1, 0.3),
        min_path_length=5,
        max_path_length=25
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
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


def generate_medium_dataset():
    """Generate a medium-sized dataset for training"""
    print("🧠 Generating medium training dataset...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=500,  # Medium dataset
        obstacle_density_range=(0.1, 0.35),
        min_path_length=5,
        max_path_length=30
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    # Save the data
    save_training_data(X_train, y_train, metadata, "data/raw/medium_training_dataset.npz")
    
    # Show summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total examples: {len(X_train)}")
    print(f"   Input shape: {X_train.shape}")
    print(f"   Output shape: {y_train.shape}")
    print(f"   Environments: {len(metadata)}")
    
    return X_train, y_train, metadata


def generate_large_dataset():
    """Generate a large dataset for full training"""
    print("🧠 Generating large training dataset...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=1000,  # Large dataset
        obstacle_density_range=(0.1, 0.4),
        min_path_length=5,
        max_path_length=50
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
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
    
    # Show first 5 examples
    for i in range(min(5, len(X_train))):
        perception = X_train[i].reshape(3, 3)
        action = y_train[i]
        
        print(f"\nExample {i+1}:")
        print(f"  3x3 Perception (flattened: {X_train[i]}):")
        for row in perception:
            print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
        print(f"  Action: {action} ({action_names[action]})")


def main():
    """Main function to generate training data"""
    parser = argparse.ArgumentParser(description='Generate training data for robot navigation')
    parser.add_argument('dataset_size', choices=['small', 'medium', 'large'], 
                       help='Size of dataset to generate')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Directory to save the dataset')
    
    args = parser.parse_args()
    
    print(f"🚀 Generating {args.dataset_size} training dataset...")
    print("=" * 60)
    
    if args.dataset_size == "small":
        X_train, y_train, metadata = generate_small_dataset()
    elif args.dataset_size == "medium":
        X_train, y_train, metadata = generate_medium_dataset()
    elif args.dataset_size == "large":
        X_train, y_train, metadata = generate_large_dataset()
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
