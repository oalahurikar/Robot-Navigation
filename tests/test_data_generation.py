"""
🧠 TEST: Data Generation System
==============================

Test script to demonstrate the complete data generation pipeline
and verify it works according to the specifications in data_generation_pipeline.md
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the project root to the path so we can import from core
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation import (
    TrainingConfig, TrainingDataGenerator, 
    visualize_environment, visualize_training_examples,
    save_training_data, load_training_data
)


def test_single_environment():
    """Test generating a single environment and extracting training data"""
    print("🧪 Testing single environment generation...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=1,
        obstacle_density_range=(0.15, 0.25),
        min_path_length=5,
        max_path_length=20
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    print(f"✅ Generated {len(X_train)} training examples")
    print(f"   Input shape: {X_train.shape}")
    print(f"   Output shape: {y_train.shape}")
    
    # Show first few examples
    print(f"\n📋 First 5 training examples:")
    for i in range(min(5, len(X_train))):
        perception = X_train[i].reshape(3, 3)
        action = y_train[i]
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        print(f"Example {i+1}:")
        print(f"  3x3 Perception:")
        for row in perception:
            print(f"    {' '.join(['X' if x > 0 else '.' for x in row])}")
        print(f"  Action: {action_names[action]}")
        print()
    
    return X_train, y_train, metadata


def test_multiple_environments():
    """Test generating multiple environments"""
    print("🧪 Testing multiple environment generation...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=50,  # Small number for quick test
        obstacle_density_range=(0.1, 0.3),
        min_path_length=5,
        max_path_length=25
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    # Analyze results
    generator.analyze_dataset(X_train, y_train, metadata)
    
    return X_train, y_train, metadata


def test_data_validation():
    """Test data validation and quality checks"""
    print("🧪 Testing data validation...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=20,
        obstacle_density_range=(0.1, 0.3),
        min_path_length=5,
        max_path_length=20
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    # Validation checks
    print(f"\n🔍 Data Validation Checks:")
    
    # Check input format
    assert X_train.shape[1] == 9, f"Expected 9 input features, got {X_train.shape[1]}"
    print("✅ Input format correct (9 features per example)")
    
    # Check output format
    unique_actions = np.unique(y_train)
    valid_actions = {0, 1, 2, 3}  # UP, DOWN, LEFT, RIGHT
    assert set(unique_actions).issubset(valid_actions), f"Invalid actions found: {unique_actions}"
    print("✅ Output format correct (valid actions only)")
    
    # Check perception values
    perception_values = np.unique(X_train)
    valid_values = {0, 1}  # Empty or obstacle
    assert set(perception_values).issubset(valid_values), f"Invalid perception values: {perception_values}"
    print("✅ Perception values correct (0 or 1 only)")
    
    # Check path lengths
    path_lengths = [m['path_length'] for m in metadata]
    assert all(config.min_path_length <= length <= config.max_path_length for length in path_lengths), \
        f"Path lengths out of range: {min(path_lengths)} - {max(path_lengths)}"
    print("✅ Path lengths within specified range")
    
    print("✅ All validation checks passed!")
    
    return X_train, y_train, metadata


def demonstrate_visualization():
    """Demonstrate visualization capabilities"""
    print("🧪 Testing visualization capabilities...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=5,
        obstacle_density_range=(0.15, 0.25),
        min_path_length=8,
        max_path_length=20
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    if metadata:
        # Visualize first environment
        sample_meta = metadata[0]
        print(f"🗺️ Visualizing environment {sample_meta['env_idx']}...")
        
        # Create a simple environment for visualization
        env = np.zeros((config.grid_size, config.grid_size))
        # Add some obstacles for demo (simplified)
        for i in range(3):
            env[2, 3+i] = 1
            env[5+i, 7] = 1
        
        visualize_environment(
            env, 
            sample_meta['start'], 
            sample_meta['goal'], 
            sample_meta['path'],
            f"Environment {sample_meta['env_idx']} - Path Length: {sample_meta['path_length']}"
        )
    
    # Visualize training examples
    print("👁️ Visualizing training examples...")
    visualize_training_examples(X_train, y_train, num_examples=6)
    
    return X_train, y_train, metadata


def test_save_load_functionality():
    """Test saving and loading training data"""
    print("🧪 Testing save/load functionality...")
    
    config = TrainingConfig(
        grid_size=10,
        num_environments=10,
        obstacle_density_range=(0.1, 0.2),
        min_path_length=5,
        max_path_length=15
    )
    
    generator = TrainingDataGenerator(config)
    X_train, y_train, metadata = generator.generate_complete_dataset()
    
    # Save data
    filename = "test_training_data.npz"
    save_training_data(X_train, y_train, metadata, filename)
    
    # Load data
    X_loaded, y_loaded, metadata_loaded = load_training_data(filename)
    
    # Verify data integrity
    assert np.array_equal(X_train, X_loaded), "Loaded X_train doesn't match saved data"
    assert np.array_equal(y_train, y_loaded), "Loaded y_train doesn't match saved data"
    assert len(metadata) == len(metadata_loaded), "Loaded metadata length doesn't match"
    
    print("✅ Save/load functionality works correctly!")
    
    return X_train, y_train, metadata


def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Running comprehensive data generation tests...")
    print("=" * 60)
    
    try:
        # Test 1: Single environment
        print("\n1️⃣ SINGLE ENVIRONMENT TEST")
        test_single_environment()
        
        # Test 2: Multiple environments
        print("\n2️⃣ MULTIPLE ENVIRONMENTS TEST")
        test_multiple_environments()
        
        # Test 3: Data validation
        print("\n3️⃣ DATA VALIDATION TEST")
        test_data_validation()
        
        # Test 4: Visualization
        print("\n4️⃣ VISUALIZATION TEST")
        demonstrate_visualization()
        
        # Test 5: Save/load
        print("\n5️⃣ SAVE/LOAD TEST")
        test_save_load_functionality()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Data generation system is working correctly")
        print("✅ Ready for neural network training!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    """Run the test suite"""
    run_comprehensive_test()
