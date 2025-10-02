"""
🧠 ROBOT NAVIGATION EVALUATION SCRIPT
=====================================

Biological Inspiration:
- Like testing if an animal can actually navigate to food sources
- Evaluates real-world navigation performance, not just action prediction

Mathematical Foundation:
- Simulates discrete navigation steps using trained models
- Measures success through goal-reaching and path efficiency

Learning Objective:
Evaluate if trained neural networks can successfully guide robots to goals
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.navigation_simulator import (
    RobotNavigationSimulator, NavigationEvaluator, 
    visualize_navigation_result, compare_navigation_modes
)
from core.data_generation import TrainingDataGenerator, TrainingConfig
from scripts.train_nn import load_data, create_model
from core.pytorch_network import load_config


def load_trained_model(model_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """Load trained neural network model"""
    model = create_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def generate_test_environments(num_envs: int = 100, 
                             grid_size: int = 10,
                             wall_padding: int = 1) -> Tuple[List[np.ndarray], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Generate test environments for navigation evaluation
    
    Args:
        num_envs: Number of test environments to generate
        grid_size: Size of navigable grid (10x10)
        wall_padding: Wall padding around navigable area
        
    Returns:
        Tuple of (environments, start_goal_pairs)
    """
    print(f"🌍 Generating {num_envs} test environments...")
    
    # Create configuration for test data generation
    test_config = TrainingConfig(
        grid_size=grid_size,
        num_environments=num_envs,
        obstacle_density_range=(0.1, 0.4),
        min_path_length=5,
        max_path_length=50,
        wall_padding=wall_padding,
        use_goal_delta=True,
        perception_size=3
    )
    
    # Generate environments
    generator = TrainingDataGenerator(test_config)
    _, _, metadata = generator.generate_complete_dataset()
    
    # Extract environments and start-goal pairs
    environments = []
    start_goals = []
    
    for meta in metadata:
        # Reconstruct environment from metadata
        # Note: We'll need to regenerate the environment since it's not stored
        # For now, we'll use the training data environments
        pass
    
    # For now, let's use the training data environments
    print("📂 Using training data environments for testing...")
    X, y, metadata, _ = load_data(use_goal_delta=True, verbose=False)
    
    # Generate environments using the same config as training
    generator = TrainingDataGenerator(test_config)
    environments = []
    start_goals = []
    
    for i in range(min(num_envs, 100)):  # Limit to 100 for testing
        try:
            env, start, goal = generator.env_generator.generate_environment()
            environments.append(env)
            start_goals.append((start, goal))
        except Exception as e:
            print(f"   Warning: Failed to generate environment {i}: {e}")
            continue
    
    print(f"✅ Generated {len(environments)} test environments")
    return environments, start_goals


def evaluate_model_performance(model_path: str,
                             use_goal_aware: bool = True,
                             num_test_envs: int = 100,
                             max_steps: int = 100,
                             verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate trained model's navigation performance
    
    Args:
        model_path: Path to trained model file
        use_goal_aware: Whether model uses goal-aware features
        num_test_envs: Number of test environments
        max_steps: Maximum navigation steps
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"🧪 Evaluating navigation performance...")
    print(f"   Model: {model_path}")
    print(f"   Mode: {'Goal-Aware' if use_goal_aware else 'Basic'}")
    print(f"   Test environments: {num_test_envs}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Load configuration
    config = load_config(goal_aware=use_goal_aware)
    
    # Load trained model
    model = load_trained_model(model_path, config, device)
    
    # Generate test environments
    environments, start_goals = generate_test_environments(
        num_envs=num_test_envs,
        grid_size=config['data']['grid_size'],
        wall_padding=config['data']['wall_padding']
    )
    
    # Create navigation simulator
    simulator = RobotNavigationSimulator(
        model=model,
        device=device,
        use_goal_aware=use_goal_aware,
        perception_size=3,
        wall_padding=config['data']['wall_padding']
    )
    
    # Create evaluator
    evaluator = NavigationEvaluator(simulator)
    
    # Run evaluation
    results = evaluator.evaluate_on_dataset(
        environments=environments,
        start_goals=start_goals,
        max_steps=max_steps,
        verbose=verbose
    )
    
    return results


def compare_models_performance(model_path_goal_aware: str,
                             model_path_basic: str,
                             num_test_envs: int = 100,
                             max_steps: int = 100) -> None:
    """
    Compare goal-aware vs basic model performance
    
    Args:
        model_path_goal_aware: Path to goal-aware model
        model_path_basic: Path to basic model
        num_test_envs: Number of test environments
        max_steps: Maximum navigation steps
    """
    print(f"🆚 Comparing model performance...")
    
    # Evaluate both models
    print(f"\n1️⃣ Evaluating Goal-Aware Model...")
    results_goal_aware = evaluate_model_performance(
        model_path=model_path_goal_aware,
        use_goal_aware=True,
        num_test_envs=num_test_envs,
        max_steps=max_steps,
        verbose=True
    )
    
    print(f"\n2️⃣ Evaluating Basic Model...")
    results_basic = evaluate_model_performance(
        model_path=model_path_basic,
        use_goal_aware=False,
        num_test_envs=num_test_envs,
        max_steps=max_steps,
        verbose=True
    )
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 50)
    
    compare_navigation_modes(
        results_goal_aware['results'],
        results_basic['results']
    )


def main():
    """Main evaluation script"""
    print("🧠 ROBOT NAVIGATION EVALUATION")
    print("=" * 50)
    
    # Example usage - you'll need to provide actual model paths
    model_path = "data/models/final_models/robot_navigation_model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please train a model first using the training script")
        return
    
    # Evaluate single model
    try:
        results = evaluate_model_performance(
            model_path=model_path,
            use_goal_aware=True,
            num_test_envs=50,  # Start with fewer environments for testing
            max_steps=100,
            verbose=True
        )
        
        # Show some example navigation visualizations
        successful_results = [r for r in results['results'] if r.success]
        failed_results = [r for r in results['results'] if not r.success]
        
        if successful_results:
            print(f"\n🎯 Showing successful navigation example...")
            visualize_navigation_result(successful_results[0])
        
        if failed_results:
            print(f"\n❌ Showing failed navigation example...")
            visualize_navigation_result(failed_results[0])
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

