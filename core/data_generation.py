"""
🧠 PROJECT: 2D Point-Robot Navigator - Data Generation Pipeline
===============================================================

Biological Inspiration: Like how animals learn navigation by observing expert 
demonstrations and memorizing state-action patterns through hippocampus place 
cells and motor cortex learning.

Mathematical Foundation: Supervised learning with expert demonstrations, where 
A* algorithm provides optimal training labels for each robot perception state.

Learning Objective: Generate training data where robot learns to navigate with 
limited 3x3 perception by imitating optimal A* pathfinding decisions.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training data generation"""
    grid_size: int = 10
    num_environments: int = 1000
    obstacle_density_range: Tuple[float, float] = (0.1, 0.4)
    min_path_length: int = 5
    max_path_length: int = 50
    max_generation_attempts: int = 100


class AStarPathfinder:
    """
    A* pathfinding algorithm for optimal path generation
    
    Biological Inspiration: Like how hippocampus place cells create
    optimal navigation paths through spatial memory
    """
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.height, self.width = grid.shape
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-connected neighbors"""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if (0 <= nx < self.height and 0 <= ny < self.width and 
                self.grid[nx, ny] == 0):
                neighbors.append((nx, ny))
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path using A* algorithm"""
        if self.grid[start] == 1 or self.grid[goal] == 1:
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: x[0])[1]
            # Remove the current item from open_set safely
            open_set = [(f, pos) for f, pos in open_set if pos != current]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    # Check if neighbor is already in open_set
                    neighbor_in_open = any(pos == neighbor for _, pos in open_set)
                    if not neighbor_in_open:
                        open_set.append((f_score[neighbor], neighbor))
        
        return None


class EnvironmentGenerator:
    """
    Generate diverse environments for training data
    
    Biological Inspiration: Like how different brain regions specialize
    in different types of spatial processing
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def generate_environment(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Generate a single environment with obstacles"""
        for attempt in range(self.config.max_generation_attempts):
            # Create empty grid
            env = np.zeros((self.config.grid_size, self.config.grid_size), dtype=int)
            
            # Place obstacles randomly
            obstacle_density = random.uniform(*self.config.obstacle_density_range)
            num_obstacles = int(obstacle_density * env.size)
            
            # Place obstacles
            positions = [(i, j) for i in range(env.shape[0]) for j in range(env.shape[1])]
            random.shuffle(positions)
            
            for i in range(min(num_obstacles, len(positions))):
                env[positions[i]] = 1
            
            # Place start and goal
            start, goal = self._place_start_goal(env)
            
            # Validate environment
            if self._validate_environment(env, start, goal):
                return env, start, goal
        
        raise RuntimeError(f"Failed to generate valid environment after {self.config.max_generation_attempts} attempts")
    
    def _place_start_goal(self, env: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Place start and goal positions"""
        empty_positions = [(i, j) for i in range(env.shape[0]) for j in range(env.shape[1]) if env[i, j] == 0]
        
        if len(empty_positions) < 2:
            raise ValueError("Not enough empty positions for start and goal")
        
        start, goal = random.sample(empty_positions, 2)
        
        # Ensure minimum distance between start and goal
        while self._manhattan_distance(start, goal) < 3:
            start, goal = random.sample(empty_positions, 2)
        
        return start, goal
    
    def _validate_environment(self, env: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Validate that environment has valid path"""
        pathfinder = AStarPathfinder(env)
        path = pathfinder.find_path(start, goal)
        
        if path is None:
            return False
        
        path_length = len(path)
        return self.config.min_path_length <= path_length <= self.config.max_path_length
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class PerceptionExtractor:
    """
    Extract robot's 3x3 perception from environment
    
    Biological Inspiration: Like how visual cortex processes
    limited peripheral vision information
    """
    
    @staticmethod
    def extract_3x3_view(env: np.ndarray, robot_pos: Tuple[int, int]) -> np.ndarray:
        """Extract 3x3 view around robot position"""
        x, y = robot_pos
        view = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                env_x = x + i - 1  # Center around robot
                env_y = y + j - 1
                
                if 0 <= env_x < env.shape[0] and 0 <= env_y < env.shape[1]:
                    view[i, j] = env[env_x, env_y]
                else:
                    view[i, j] = 1  # Treat out-of-bounds as obstacles
        
        return view
    
    @staticmethod
    def movement_to_action(current_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> int:
        """Convert movement to discrete action"""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        if dx == -1: return 0    # UP
        elif dx == 1: return 1   # DOWN
        elif dy == -1: return 2  # LEFT
        elif dy == 1: return 3   # RIGHT
        else: return 4           # STAY (should not happen with A*)


class TrainingDataGenerator:
    """
    Main class for generating complete training dataset
    
    Biological Inspiration: Like how the brain creates comprehensive
    learning experiences through systematic exploration
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env_generator = EnvironmentGenerator(config)
        self.perception_extractor = PerceptionExtractor()
        
    def generate_complete_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate complete training dataset for robot navigation
        
        Returns:
        X_train: (n_examples, 9) - Robot 3x3 perceptions
        y_train: (n_examples,) - Optimal actions from A*
        metadata: List of environment metadata
        """
        
        all_perceptions = []
        all_actions = []
        all_metadata = []
        
        print(f"🧠 Generating training data for {self.config.num_environments} environments...")
        
        for env_idx in range(self.config.num_environments):
            if (env_idx + 1) % 100 == 0:
                print(f"   Progress: {env_idx + 1}/{self.config.num_environments} environments")
            
            try:
                # Step 1: Generate 10x10 environment
                env_10x10, start, goal = self.env_generator.generate_environment()
                
                # Step 2: Find A* optimal path
                pathfinder = AStarPathfinder(env_10x10)
                a_star_path = pathfinder.find_path(start, goal)
                
                if a_star_path is None:
                    continue
                
                # Step 3: Extract training examples from path
                env_perceptions = []
                env_actions = []
                
                for i in range(len(a_star_path) - 1):
                    current_pos = a_star_path[i]
                    next_pos = a_star_path[i + 1]
                    
                    # Extract 3x3 perception around current position
                    perception_3x3 = self.perception_extractor.extract_3x3_view(env_10x10, current_pos)
                    flattened_perception = perception_3x3.flatten()
                    
                    # Convert movement to action
                    action = self.perception_extractor.movement_to_action(current_pos, next_pos)
                    
                    env_perceptions.append(flattened_perception)
                    env_actions.append(action)
                
                # Add to complete dataset
                all_perceptions.extend(env_perceptions)
                all_actions.extend(env_actions)
                
                # Store metadata
                metadata = {
                    'env_idx': env_idx,
                    'start': start,
                    'goal': goal,
                    'path_length': len(a_star_path),
                    'obstacle_count': np.sum(env_10x10 == 1),
                    'obstacle_density': np.sum(env_10x10 == 1) / env_10x10.size,
                    'path': a_star_path
                }
                all_metadata.append(metadata)
                
            except Exception as e:
                print(f"   Warning: Failed to generate environment {env_idx}: {e}")
                continue
        
        print(f"✅ Generated {len(all_perceptions)} training examples from {len(all_metadata)} environments")
        
        # Memory optimization: Use float32 for inputs (50% memory savings) and int8 for labels (87.5% savings)
        X_train = np.array(all_perceptions, dtype=np.float32)  # Binary values (0.0/1.0) work fine with float32
        y_train = np.array(all_actions, dtype=np.int8)         # Only 4 actions (0-3) fit easily in int8
        
        # Memory usage comparison:
        # Before: float64 (8 bytes) + int64 (8 bytes) = 16 bytes per sample
        # After:  float32 (4 bytes) + int8 (1 byte) = 5 bytes per sample (69% savings!)
        
        return X_train, y_train, all_metadata
    
    def analyze_dataset(self, X_train: np.ndarray, y_train: np.ndarray, metadata: List[Dict]) -> None:
        """Analyze generated dataset"""
        print("\n📊 DATASET ANALYSIS:")
        print("=" * 50)
        
        # Basic statistics
        print(f"Total training examples: {len(X_train)}")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        print(f"Input data type: {X_train.dtype}")
        print(f"Output data type: {y_train.dtype}")
        
        # Memory usage
        input_memory_mb = X_train.nbytes / 1024 / 1024
        output_memory_mb = y_train.nbytes / 1024 / 1024
        total_memory_mb = input_memory_mb + output_memory_mb
        print(f"Memory usage: {total_memory_mb:.3f} MB (Input: {input_memory_mb:.3f} MB, Output: {output_memory_mb:.3f} MB)")
        
        # Action distribution
        action_counts = np.bincount(y_train.astype(int))
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"\nAction distribution:")
        for i, (count, name) in enumerate(zip(action_counts, action_names)):
            percentage = (count / len(y_train)) * 100
            print(f"  {name}: {count} ({percentage:.1f}%)")
        
        # Environment statistics
        path_lengths = [m['path_length'] for m in metadata]
        obstacle_densities = [m['obstacle_density'] for m in metadata]
        
        print(f"\nEnvironment statistics:")
        print(f"  Average path length: {np.mean(path_lengths):.1f}")
        print(f"  Path length range: {min(path_lengths)} - {max(path_lengths)}")
        print(f"  Average obstacle density: {np.mean(obstacle_densities):.1%}")
        print(f"  Obstacle density range: {min(obstacle_densities):.1%} - {max(obstacle_densities):.1%}")
        
        # Perception statistics
        perception_stats = {
            'mean_obstacles_per_view': np.mean(np.sum(X_train, axis=1)),
            'max_obstacles_per_view': np.max(np.sum(X_train, axis=1)),
            'min_obstacles_per_view': np.min(np.sum(X_train, axis=1))
        }
        
        print(f"\nPerception statistics:")
        print(f"  Average obstacles per 3x3 view: {perception_stats['mean_obstacles_per_view']:.2f}")
        print(f"  Max obstacles per view: {perception_stats['max_obstacles_per_view']}")
        print(f"  Min obstacles per view: {perception_stats['min_obstacles_per_view']}")


def visualize_environment(env: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], 
                         path: List[Tuple[int, int]], title: str = "Sample Environment") -> None:
    """Visualize a sample environment with path"""
    
    # Create visualization
    vis_env = env.copy().astype(float)
    vis_env[start] = 0.5  # Start position
    vis_env[goal] = 0.8   # Goal position
    
    # Mark path
    for pos in path[1:-1]:  # Skip start and goal
        vis_env[pos] = 0.3
    
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_env, cmap='viridis')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Add text annotations
    plt.text(start[1], start[0], 'R', ha='center', va='center', fontsize=16, color='white', weight='bold')
    plt.text(goal[1], goal[0], 'G', ha='center', va='center', fontsize=16, color='white', weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Cell Type')
    
    plt.show()


def visualize_training_examples(X_train: np.ndarray, y_train: np.ndarray, num_examples: int = 9) -> None:
    """Visualize sample training examples"""
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(num_examples, len(X_train))):
        # Reshape flattened perception back to 3x3
        perception = X_train[i].reshape(3, 3)
        action = y_train[i]
        
        axes[i].imshow(perception, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Action: {action_names[action]}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        
        # Add grid
        axes[i].set_xticks(range(3))
        axes[i].set_yticks(range(3))
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_examples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def save_training_data(X_train: np.ndarray, y_train: np.ndarray, metadata: List[Dict], 
                      filename: str = "training_data.npz") -> None:
    """Save training data to file"""
    
    # Convert metadata to arrays for saving
    env_indices = [m['env_idx'] for m in metadata]
    path_lengths = [m['path_length'] for m in metadata]
    obstacle_counts = [m['obstacle_count'] for m in metadata]
    obstacle_densities = [m['obstacle_density'] for m in metadata]
    
    np.savez(filename,
             X_train=X_train,
             y_train=y_train,
             env_indices=env_indices,
             path_lengths=path_lengths,
             obstacle_counts=obstacle_counts,
             obstacle_densities=obstacle_densities)
    
    print(f"💾 Training data saved to {filename}")


def load_training_data(filename: str = "training_data.npz") -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load training data from file"""
    
    data = np.load(filename)
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Reconstruct metadata
    metadata = []
    for i in range(len(data['env_indices'])):
        metadata.append({
            'env_idx': data['env_indices'][i],
            'path_length': data['path_lengths'][i],
            'obstacle_count': data['obstacle_counts'][i],
            'obstacle_density': data['obstacle_densities'][i]
        })
    
    print(f"📂 Training data loaded from {filename}")
    return X_train, y_train, metadata
