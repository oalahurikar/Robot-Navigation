"""
🧠 ROBOT NAVIGATION SIMULATOR
=============================

Biological Inspiration:
- Like how animals navigate using learned spatial maps and goal-seeking behavior
- Tests if the neural network can actually guide a robot to destinations

Mathematical Foundation:
- Simulates discrete navigation steps using trained model predictions
- Measures success through goal-reaching and path efficiency metrics

Learning Objective:
Evaluate if trained neural network can successfully navigate robots to goals
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import our existing classes
from .data_generation import AStarPathfinder, PerceptionExtractor, TrainingConfig


class Action(Enum):
    """Navigation actions"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class NavigationResult:
    """Result of robot navigation simulation"""
    success: bool
    path: List[Tuple[int, int]]
    steps_taken: int
    collisions: int
    final_distance_to_goal: float
    optimal_path_length: int
    path_efficiency: float
    environment: np.ndarray
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    use_goal_aware: bool


class RobotNavigationSimulator:
    """
    Simulate robot navigation using trained neural network
    
    Biological Inspiration: Like testing if an animal can actually
    navigate to a food source using learned spatial memories
    """
    
    def __init__(self, 
                 model, 
                 device: torch.device,
                 use_goal_aware: bool = True,
                 perception_size: int = 3,
                 wall_padding: int = 1):
        """
        Initialize navigation simulator
        
        Args:
            model: Trained neural network model
            device: PyTorch device (CPU/GPU)
            use_goal_aware: Whether to use goal delta features
            perception_size: Size of perception window (3 for 3×3)
            wall_padding: Wall padding around navigable area
        """
        self.model = model
        self.device = device
        self.use_goal_aware = use_goal_aware
        self.perception_size = perception_size
        self.wall_padding = wall_padding
        
        # Initialize perception extractor
        self.perception_extractor = PerceptionExtractor(
            perception_size=perception_size,
            use_goal_delta=use_goal_aware
        )
        
        # Action mapping
        self.action_deltas = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
    
    def simulate_navigation(self,
                          environment: np.ndarray,
                          start_pos: Tuple[int, int],
                          goal_pos: Tuple[int, int],
                          max_steps: int = 100,
                          verbose: bool = False) -> NavigationResult:
        """
        Simulate robot navigation from start to goal
        
        Args:
            environment: 2D grid environment with obstacles
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            max_steps: Maximum navigation steps
            verbose: Whether to print progress
            
        Returns:
            NavigationResult with success status and metrics
        """
        self.model.eval()
        
        # Initialize navigation state
        current_pos = start_pos
        path = [current_pos]
        collisions = 0
        steps_taken = 0
        
        # Calculate optimal path for comparison
        pathfinder = AStarPathfinder(environment)
        optimal_path = pathfinder.find_path(start_pos, goal_pos)
        optimal_path_length = len(optimal_path) if optimal_path else float('inf')
        
        if verbose:
            print(f"🚀 Starting navigation from {start_pos} to {goal_pos}")
            print(f"📏 Optimal path length: {optimal_path_length}")
        
        # Navigation loop
        for step in range(max_steps):
            steps_taken = step + 1
            
            # Check if goal reached
            if current_pos == goal_pos:
                success = True
                final_distance = 0.0
                break
            
            # Extract perception and predict action
            try:
                perception = self.perception_extractor.extract_goal_aware_perception(
                    environment, current_pos, goal_pos
                )
                
                # Convert to tensor and predict
                perception_tensor = torch.FloatTensor(perception).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(perception_tensor)
                    predicted_action = torch.argmax(outputs, dim=1).item()
                
                # Convert to Action enum
                action = Action(predicted_action)
                
                # Calculate next position
                delta = self.action_deltas[action]
                next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                
                # Check for collision
                if self._is_collision(environment, next_pos):
                    collisions += 1
                    if verbose:
                        print(f"   Step {step+1}: Collision at {next_pos}")
                    # Stay in current position on collision
                    next_pos = current_pos
                
                # Update position
                current_pos = next_pos
                path.append(current_pos)
                
                if verbose and (step + 1) % 10 == 0:
                    distance = self._manhattan_distance(current_pos, goal_pos)
                    print(f"   Step {step+1}: Position {current_pos}, Distance to goal: {distance}")
                    
            except Exception as e:
                if verbose:
                    print(f"   Error at step {step+1}: {e}")
                collisions += 1
                break
        
        else:
            # Max steps reached without reaching goal
            success = False
        
        # Calculate final metrics
        final_distance = self._manhattan_distance(current_pos, goal_pos)
        path_efficiency = optimal_path_length / len(path) if optimal_path_length > 0 else 0.0
        
        if verbose:
            print(f"✅ Navigation complete: {'SUCCESS' if success else 'FAILED'}")
            print(f"   Steps taken: {steps_taken}")
            print(f"   Collisions: {collisions}")
            print(f"   Final distance to goal: {final_distance}")
            print(f"   Path efficiency: {path_efficiency:.2f}")
        
        return NavigationResult(
            success=success,
            path=path,
            steps_taken=steps_taken,
            collisions=collisions,
            final_distance_to_goal=final_distance,
            optimal_path_length=optimal_path_length,
            path_efficiency=path_efficiency,
            environment=environment.copy(),
            start_pos=start_pos,
            goal_pos=goal_pos,
            use_goal_aware=self.use_goal_aware
        )
    
    def _is_collision(self, environment: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if position is a collision (wall or obstacle)"""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= environment.shape[0] or y < 0 or y >= environment.shape[1]:
            return True
        
        # Check for obstacles
        return environment[x, y] == 1
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class NavigationEvaluator:
    """
    Evaluate robot navigation performance across multiple environments
    
    Biological Inspiration: Like testing navigation skills across
    different environments to measure learning generalization
    """
    
    def __init__(self, simulator: RobotNavigationSimulator):
        """Initialize evaluator with navigation simulator"""
        self.simulator = simulator
    
    def evaluate_on_dataset(self,
                           environments: List[np.ndarray],
                           start_goals: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                           max_steps: int = 100,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate navigation performance on multiple environments
        
        Args:
            environments: List of environment grids
            start_goals: List of (start_pos, goal_pos) tuples
            max_steps: Maximum steps per navigation
            verbose: Whether to print progress
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if len(environments) != len(start_goals):
            raise ValueError("Number of environments must match number of start-goal pairs")
        
        results = []
        successful_navigations = 0
        
        if verbose:
            print(f"🧪 Evaluating navigation on {len(environments)} environments...")
        
        for i, (env, (start, goal)) in enumerate(zip(environments, start_goals)):
            if verbose and (i + 1) % 50 == 0:
                print(f"   Progress: {i + 1}/{len(environments)} environments")
            
            result = self.simulator.simulate_navigation(
                environment=env,
                start_pos=start,
                goal_pos=goal,
                max_steps=max_steps,
                verbose=False
            )
            
            results.append(result)
            if result.success:
                successful_navigations += 1
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(results)
        
        if verbose:
            print(f"✅ Evaluation complete!")
            print(f"   Success rate: {metrics['success_rate']:.1%}")
            print(f"   Average steps: {metrics['avg_steps']:.1f}")
            print(f"   Average path efficiency: {metrics['avg_path_efficiency']:.2f}")
            print(f"   Average collisions: {metrics['avg_collisions']:.1f}")
        
        return {
            'results': results,
            'metrics': metrics,
            'total_environments': len(environments),
            'total_navigations': len(results),
            'successful_navigations': successful_navigations
        }
    
    def _calculate_metrics(self, results: List[NavigationResult]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        if not results:
            return {}
        
        # Success metrics
        success_count = sum(1 for r in results if r.success)
        success_rate = success_count / len(results)
        
        # Path efficiency (only for successful navigations)
        successful_results = [r for r in results if r.success]
        if successful_results:
            avg_path_efficiency = np.mean([r.path_efficiency for r in successful_results])
            avg_steps = np.mean([r.steps_taken for r in successful_results])
            avg_collisions = np.mean([r.collisions for r in successful_results])
        else:
            avg_path_efficiency = 0.0
            avg_steps = float('inf')
            avg_collisions = np.mean([r.collisions for r in results])
        
        # Distance metrics (for all results)
        avg_final_distance = np.mean([r.final_distance_to_goal for r in results])
        min_final_distance = min([r.final_distance_to_goal for r in results])
        max_final_distance = max([r.final_distance_to_goal for r in results])
        
        return {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_path_efficiency': avg_path_efficiency,
            'avg_collisions': avg_collisions,
            'avg_final_distance': avg_final_distance,
            'min_final_distance': min_final_distance,
            'max_final_distance': max_final_distance,
            'successful_navigations': success_count,
            'total_navigations': len(results)
        }


def visualize_navigation_result(result: NavigationResult, 
                              save_path: Optional[str] = None) -> None:
    """
    Visualize robot navigation path vs optimal path
    
    Biological Inspiration: Like visualizing the actual vs optimal
    path taken by an animal to understand navigation efficiency
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Environment with actual path
    env = result.environment.copy()
    
    # Mark path
    for i, pos in enumerate(result.path):
        if i == 0:
            env[pos] = 3  # Start position
        elif i == len(result.path) - 1:
            env[pos] = 4  # End position
        else:
            env[pos] = 2  # Path
    
    # Mark start and goal
    env[result.start_pos] = 3
    env[result.goal_pos] = 4
    
    im1 = ax1.imshow(env, cmap='viridis')
    ax1.set_title(f'Robot Navigation Path\n{"✅ SUCCESS" if result.success else "❌ FAILED"}')
    ax1.set_xlabel('Y Coordinate')
    ax1.set_ylabel('X Coordinate')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='black', label='Walls/Obstacles'),
        plt.Rectangle((0,0),1,1, facecolor='lightgray', label='Empty Space'),
        plt.Rectangle((0,0),1,1, facecolor='yellow', label='Robot Path'),
        plt.Rectangle((0,0),1,1, facecolor='green', label='Start'),
        plt.Rectangle((0,0),1,1, facecolor='red', label='Goal')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Optimal A* path for comparison
    pathfinder = AStarPathfinder(result.environment)
    optimal_path = pathfinder.find_path(result.start_pos, result.goal_pos)
    
    env_optimal = result.environment.copy()
    if optimal_path:
        for i, pos in enumerate(optimal_path):
            if i == 0:
                env_optimal[pos] = 3  # Start
            elif i == len(optimal_path) - 1:
                env_optimal[pos] = 4  # Goal
            else:
                env_optimal[pos] = 2  # Optimal path
    
    im2 = ax2.imshow(env_optimal, cmap='viridis')
    ax2.set_title(f'Optimal A* Path\nLength: {result.optimal_path_length}')
    ax2.set_xlabel('Y Coordinate')
    ax2.set_ylabel('X Coordinate')
    
    # Add metrics text
    metrics_text = f"""Navigation Metrics:
Steps Taken: {result.steps_taken}
Collisions: {result.collisions}
Path Efficiency: {result.path_efficiency:.2f}
Final Distance: {result.final_distance_to_goal}"""
    
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Navigation visualization saved to {save_path}")
    
    plt.show()


def compare_navigation_modes(results_goal_aware: List[NavigationResult],
                           results_basic: List[NavigationResult]) -> None:
    """
    Compare goal-aware vs basic navigation performance
    
    Biological Inspiration: Like comparing navigation performance
    with and without compass-like directional awareness
    """
    if not results_goal_aware or not results_basic:
        print("❌ Cannot compare: One or both result lists are empty")
        return
    
    # Calculate metrics for both modes
    evaluator = NavigationEvaluator(None)  # We only need the metrics calculation
    
    metrics_goal_aware = evaluator._calculate_metrics(results_goal_aware)
    metrics_basic = evaluator._calculate_metrics(results_basic)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Success rate comparison
    modes = ['Basic (9 features)', 'Goal-Aware (11 features)']
    success_rates = [metrics_basic['success_rate'], metrics_goal_aware['success_rate']]
    
    bars1 = axes[0, 0].bar(modes, success_rates, color=['lightcoral', 'lightblue'])
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim(0, 1)
    
    # Add percentage labels
    for bar, rate in zip(bars1, success_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{rate:.1%}', ha='center', va='bottom')
    
    # Path efficiency comparison
    path_efficiencies = [metrics_basic['avg_path_efficiency'], metrics_goal_aware['avg_path_efficiency']]
    
    bars2 = axes[0, 1].bar(modes, path_efficiencies, color=['lightcoral', 'lightblue'])
    axes[0, 1].set_title('Average Path Efficiency')
    axes[0, 1].set_ylabel('Path Efficiency')
    
    for bar, eff in zip(bars2, path_efficiencies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{eff:.2f}', ha='center', va='bottom')
    
    # Average steps comparison
    avg_steps = [metrics_basic['avg_steps'], metrics_goal_aware['avg_steps']]
    
    bars3 = axes[1, 0].bar(modes, avg_steps, color=['lightcoral', 'lightblue'])
    axes[1, 0].set_title('Average Steps to Goal')
    axes[1, 0].set_ylabel('Steps')
    
    for bar, steps in zip(bars3, avg_steps):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{steps:.1f}', ha='center', va='bottom')
    
    # Collisions comparison
    avg_collisions = [metrics_basic['avg_collisions'], metrics_goal_aware['avg_collisions']]
    
    bars4 = axes[1, 1].bar(modes, avg_collisions, color=['lightcoral', 'lightblue'])
    axes[1, 1].set_title('Average Collisions')
    axes[1, 1].set_ylabel('Collisions')
    
    for bar, coll in zip(bars4, avg_collisions):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{coll:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison
    print(f"\n📊 NAVIGATION MODE COMPARISON")
    print("=" * 50)
    print(f"Success Rate:")
    print(f"   Basic:      {metrics_basic['success_rate']:.1%}")
    print(f"   Goal-Aware: {metrics_goal_aware['success_rate']:.1%}")
    print(f"   Improvement: {((metrics_goal_aware['success_rate'] - metrics_basic['success_rate']) / metrics_basic['success_rate'] * 100):+.1f}%")
    
    print(f"\nPath Efficiency:")
    print(f"   Basic:      {metrics_basic['avg_path_efficiency']:.2f}")
    print(f"   Goal-Aware: {metrics_goal_aware['avg_path_efficiency']:.2f}")
    print(f"   Improvement: {((metrics_goal_aware['avg_path_efficiency'] - metrics_basic['avg_path_efficiency']) / metrics_basic['avg_path_efficiency'] * 100):+.1f}%")
    
    print(f"\nAverage Steps:")
    print(f"   Basic:      {metrics_basic['avg_steps']:.1f}")
    print(f"   Goal-Aware: {metrics_goal_aware['avg_steps']:.1f}")
    
    print(f"\nAverage Collisions:")
    print(f"   Basic:      {metrics_basic['avg_collisions']:.1f}")
    print(f"   Goal-Aware: {metrics_goal_aware['avg_collisions']:.1f}")

