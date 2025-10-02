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


def generate_small_dataset(use_enhanced=True, perception_size=3, use_distance=False, max_distance=5):
    """Generate a small dataset for quick testing"""
    perception_desc = "3×3" if perception_size == 3 else "5×5"
    mode_desc = "distance-based" if use_distance else "binary"
    print(f"🧠 Generating small training dataset with {perception_desc} {mode_desc} perception...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=100,  # Small dataset for quick generation
        obstacle_density_range=(0.1, 0.3),
        min_path_length=5,
        max_path_length=25,
        history_length=3,  # Solution 1: Memory/History
        perception_size=perception_size,  # 3×3 or 5×5 perception
        use_distance_field=use_distance,  # Solution 3: Distance-based perception
        max_sensing_distance=max_distance
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


def generate_medium_dataset(use_enhanced=True, perception_size=3, use_distance=False, max_distance=5):
    """Generate a medium-sized dataset for training"""
    perception_desc = "3×3" if perception_size == 3 else "5×5"
    mode_desc = "distance-based" if use_distance else "binary"
    print(f"🧠 Generating medium training dataset with {perception_desc} {mode_desc} perception...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=500,  # Medium dataset
        obstacle_density_range=(0.1, 0.35),
        min_path_length=5,
        max_path_length=30,
        history_length=3,  # Solution 1: Memory/History
        perception_size=perception_size,  # 3×3 or 5×5 perception
        use_distance_field=use_distance,  # Solution 3: Distance-based perception
        max_sensing_distance=max_distance
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


def generate_large_dataset(use_enhanced=True, perception_size=3, use_distance=False, max_distance=5):
    """Generate a large dataset for full training"""
    perception_desc = "3×3" if perception_size == 3 else "5×5"
    mode_desc = "distance-based" if use_distance else "binary"
    print(f"🧠 Generating large training dataset with {perception_desc} {mode_desc} perception...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=1000,  # Large dataset
        obstacle_density_range=(0.1, 0.4),
        min_path_length=5,
        max_path_length=50,
        history_length=3,  # Solution 1: Memory/History
        perception_size=perception_size,  # 3×3 or 5×5 perception
        use_distance_field=use_distance,  # Solution 3: Distance-based perception
        max_sensing_distance=max_distance
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
    
    # Detect if using distance-based perception (continuous values between 0 and 1)
    # vs binary (only 0.0 and 1.0 values)
    sample_perception = X_train[0][:25] if feature_size >= 25 else X_train[0][:9]
    unique_vals = np.unique(sample_perception)
    is_distance_based = len(unique_vals) > 2 or (len(unique_vals) == 2 and not (0.0 in unique_vals and 1.0 in unique_vals))
    
    # Show first 5 examples
    for i in range(min(5, len(X_train))):
        action = y_train[i]
        
        print(f"\nExample {i+1}:")
        
        if feature_size == 37:
            # Enhanced 5×5 mode: 25 perception + 12 history
            perception = X_train[i][:25].reshape(5, 5)
            history = X_train[i][25:37].reshape(3, 4)  # 3 actions × 4 one-hot
            
            print(f"  5×5 Perception {'(Distance)' if is_distance_based else '(Binary)'}:")
            for row in perception:
                if is_distance_based:
                    # Show distance values
                    print(f"    {' '.join([f'{x:.1f}' for x in row])}")
                else:
                    # Show binary (X for obstacle, . for free)
                    print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
            
            print(f"  Action History (last 3 actions):")
            for j, action_vec in enumerate(history):
                action_idx = np.argmax(action_vec) if np.sum(action_vec) > 0 else -1
                if action_idx >= 0:
                    print(f"    {j+1}. {action_names[action_idx]}")
                else:
                    print(f"    {j+1}. (no action)")
        elif feature_size == 25:
            # Basic 5×5 mode: 25 perception only
            perception = X_train[i].reshape(5, 5)
            print(f"  5×5 Perception {'(Distance)' if is_distance_based else '(Binary)'}:")
            for row in perception:
                if is_distance_based:
                    print(f"    {' '.join([f'{x:.1f}' for x in row])}")
                else:
                    print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
        elif feature_size == 21:
            # Enhanced 3×3 mode: 9 perception + 12 history
            perception = X_train[i][:9].reshape(3, 3)
            history = X_train[i][9:21].reshape(3, 4)  # 3 actions × 4 one-hot
            
            print(f"  3×3 Perception {'(Distance)' if is_distance_based else '(Binary)'}:")
            for row in perception:
                if is_distance_based:
                    print(f"    {' '.join([f'{x:.1f}' for x in row])}")
                else:
                    print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
            
            print(f"  Action History (last 3 actions):")
            for j, action_vec in enumerate(history):
                action_idx = np.argmax(action_vec) if np.sum(action_vec) > 0 else -1
                if action_idx >= 0:
                    print(f"    {j+1}. {action_names[action_idx]}")
                else:
                    print(f"    {j+1}. (no action)")
        elif feature_size == 9:
            # Basic 3×3 mode: 9 perception only
            perception = X_train[i].reshape(3, 3)
            print(f"  3×3 Perception {'(Distance)' if is_distance_based else '(Binary)'}:")
            for row in perception:
                if is_distance_based:
                    print(f"    {' '.join([f'{x:.1f}' for x in row])}")
                else:
                    print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
        else:
            print(f"  Unknown feature format: {feature_size} features")
            continue
        
        print(f"  → Action: {action} ({action_names[action]})")


def main():
    """Main function to generate training data"""
    parser = argparse.ArgumentParser(description='Generate training data for robot navigation')
    parser.add_argument('dataset_size', choices=['small', 'medium', 'large'], 
                       help='Size of dataset to generate')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Directory to save the dataset')
    parser.add_argument('--basic', action='store_true', 
                       help='Use basic mode (9 features) instead of enhanced mode')
    parser.add_argument('--perception', choices=['3x3', '5x5'], default='5x5',
                       help='Perception window size (3x3 or 5x5)')
    parser.add_argument('--distance', action='store_true',
                       help='Use distance-based perception (Solution 3) instead of binary')
    parser.add_argument('--max-distance', type=int, default=5,
                       help='Maximum sensing distance for normalization (default: 5)')
    
    args = parser.parse_args()
    
    use_enhanced = not args.basic
    perception_size = 5 if args.perception == '5x5' else 3
    use_distance = args.distance
    max_distance = args.max_distance
    
    # Calculate feature count
    perception_features = perception_size * perception_size
    history_features = 12 if use_enhanced else 0
    total_features = perception_features + history_features
    
    # Build description string
    perception_type = "distance-based" if use_distance else "binary"
    if args.basic:
        mode_str = f"basic ({perception_features} {perception_type} features)"
    else:
        mode_str = f"enhanced ({total_features} features: {perception_features} {perception_type} perception + {history_features} history)"
    
    print(f"🚀 Generating {args.dataset_size} training dataset ({mode_str})...")
    print("=" * 60)
    
    if args.dataset_size == "small":
        X_train, y_train, metadata = generate_small_dataset(
            use_enhanced=use_enhanced, perception_size=perception_size, 
            use_distance=use_distance, max_distance=max_distance
        )
    elif args.dataset_size == "medium":
        X_train, y_train, metadata = generate_medium_dataset(
            use_enhanced=use_enhanced, perception_size=perception_size,
            use_distance=use_distance, max_distance=max_distance
        )
    elif args.dataset_size == "large":
        X_train, y_train, metadata = generate_large_dataset(
            use_enhanced=use_enhanced, perception_size=perception_size,
            use_distance=use_distance, max_distance=max_distance
        )
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
