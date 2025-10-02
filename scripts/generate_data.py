#!/usr/bin/env python3
"""
🧠 Robot Navigation Data Generation Script
==========================================

Generate training datasets for goal-aware robot navigation.

Usage:
    python scripts/generate_data.py small    # 100 environments
    python scripts/generate_data.py medium   # 500 environments  
    python scripts/generate_data.py large    # 1000 environments
    python scripts/generate_data.py large --basic  # Basic mode (9 features)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation import TrainingDataGenerator, TrainingConfig, save_training_data


def generate_small_dataset(use_goal_delta=True):
    """Generate a small dataset for quick testing"""
    mode_desc = "goal-aware" if use_goal_delta else "basic"
    print(f"🧠 Generating small training dataset with {mode_desc} perception...")
    
    config = TrainingConfig(
        grid_size=10,
        wall_padding=1,
        num_environments=100,  # Small dataset for quick generation
        obstacle_density_range=(0.1, 0.3),
        min_path_length=5,
        max_path_length=25,
        perception_size=3,  # 3×3 perception
        use_goal_delta=use_goal_delta
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    # Save dataset
    filename = "small_training_dataset_basic.npz" if not use_goal_delta else "small_training_dataset.npz"
    save_training_data(X_train, y_train, metadata, filename)
    
    # Show analysis
    generator.analyze_dataset(X_train, y_train, metadata)
    show_sample_data(X_train, y_train, 3)
    
    return X_train, y_train, metadata


def generate_medium_dataset(use_goal_delta=True):
    """Generate a medium-sized dataset for training"""
    mode_desc = "goal-aware" if use_goal_delta else "basic"
    print(f"🧠 Generating medium training dataset with {mode_desc} perception...")
    
    config = TrainingConfig(
        grid_size=10,
        wall_padding=1,
        num_environments=500,  # Medium dataset
        obstacle_density_range=(0.1, 0.35),
        min_path_length=5,
        max_path_length=30,
        perception_size=3,  # 3×3 perception
        use_goal_delta=use_goal_delta
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    # Save dataset
    filename = "medium_training_dataset_basic.npz" if not use_goal_delta else "medium_training_dataset.npz"
    save_training_data(X_train, y_train, metadata, filename)
    
    # Show analysis
    generator.analyze_dataset(X_train, y_train, metadata)
    show_sample_data(X_train, y_train, 3)
    
    return X_train, y_train, metadata


def generate_large_dataset(use_goal_delta=True):
    """Generate a large dataset for full training"""
    mode_desc = "goal-aware" if use_goal_delta else "basic"
    print(f"🧠 Generating large training dataset with {mode_desc} perception...")
    
    config = TrainingConfig(
        grid_size=10,
        wall_padding=1,
        num_environments=1000,  # Large dataset for full training
        obstacle_density_range=(0.1, 0.4),
        min_path_length=5,
        max_path_length=50,
        perception_size=3,  # 3×3 perception
        use_goal_delta=use_goal_delta
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    # Save dataset
    filename = "large_training_dataset_basic.npz" if not use_goal_delta else "large_training_dataset.npz"
    save_training_data(X_train, y_train, metadata, filename)
    
    # Show analysis
    generator.analyze_dataset(X_train, y_train, metadata)
    show_sample_data(X_train, y_train, 3)
    
    return X_train, y_train, metadata


def show_sample_data(X_train, y_train, num_samples=3):
    """Show sample training data"""
    print(f"\n👁️  SAMPLE TRAINING DATA:")
    print("=" * 60)
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    feature_count = X_train.shape[1]
    
    for i in range(min(num_samples, len(X_train))):
        print(f"\nExample {i+1}:")
        sample_x = X_train[i]
        sample_y = y_train[i]
        
        if feature_count == 11:
            # Goal-aware mode: 9 perception + 2 goal_delta
            perception = sample_x[:9].reshape(3, 3)
            goal_delta = sample_x[9:11]
            
            print(f"  3×3 Perception (Binary):")
            for row in perception:
                print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
            
            print(f"  Goal Delta (dx, dy): ({goal_delta[0]:.0f}, {goal_delta[1]:.0f})")
            print(f"    → Direction: {'UP' if goal_delta[0] < 0 else 'DOWN'} {'LEFT' if goal_delta[1] < 0 else 'RIGHT'}")
            print(f"    → Distance: {abs(goal_delta[0]) + abs(goal_delta[1])} steps")
            
        elif feature_count == 9:
            # Basic mode: 9 perception only
            perception = sample_x.reshape(3, 3)
            print(f"  3×3 Perception (Binary):")
            for row in perception:
                print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
        
        print(f"  → Action: {action_names[sample_y]}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate robot navigation training data')
    parser.add_argument('dataset_size', choices=['small', 'medium', 'large'],
                       help='Size of dataset to generate')
    parser.add_argument('--basic', action='store_true', 
                       help='Use basic mode (9 features) instead of goal-aware mode')
    
    args = parser.parse_args()
    
    # Determine mode
    use_goal_delta = not args.basic
    mode_type = "Goal-Aware" if use_goal_delta else "Basic"
    
    print(f"🎯 ROBOT NAVIGATION DATA GENERATION")
    print("=" * 50)
    print(f"   Dataset: {args.dataset_size}")
    print(f"   Mode: {mode_type}")
    
    # Feature count calculation
    perception_features = 9  # 3×3 perception
    goal_features = 2 if use_goal_delta else 0  # goal_delta (dx, dy)
    total_features = perception_features + goal_features
    
    print(f"   Features: {total_features} ({perception_features} perception + {goal_features} goal_delta)")
    print(f"   Architecture: {total_features} → 64 → 32 → 4")
    print()
    
    # Generate dataset
    if args.dataset_size == "small":
        X_train, y_train, metadata = generate_small_dataset(use_goal_delta=use_goal_delta)
    elif args.dataset_size == "medium":
        X_train, y_train, metadata = generate_medium_dataset(use_goal_delta=use_goal_delta)
    elif args.dataset_size == "large":
        X_train, y_train, metadata = generate_large_dataset(use_goal_delta=use_goal_delta)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"   Ready for training with: python scripts/train_nn.py{' --basic' if args.basic else ''}")


if __name__ == "__main__":
    main()