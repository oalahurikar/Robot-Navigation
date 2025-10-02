"""
🧠 PROJECT: 2D Point-Robot Navigator - Data Generation Pipeline
===============================================================

Biological Inspiration: Like how animals learn navigation by observing expert 
demonstrations and using goal-relative spatial awareness (like a compass).

Mathematical Foundation: Supervised learning with expert demonstrations, where 
A* algorithm provides optimal training labels for each robot perception state.

Learning Objective: Generate training data where robot learns to navigate with 
limited 3x3 perception + goal direction by imitating optimal A* pathfinding decisions.
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
    grid_size: int = 10  # Inner navigable area (10x10)
    wall_padding: int = 1  # Wall border around navigable area
    num_environments: int = 1000
    obstacle_density_range: Tuple[float, float] = (0.1, 0.4)
    min_path_length: int = 5
    max_path_length: int = 50
    max_generation_attempts: int = 100
    perception_size: int = 3  # Perception window size (3x3 only for original solution)
    use_goal_delta: bool = True  # Include goal relative coordinates (dx, dy)


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
        """Generate a single environment with obstacles and wall padding"""
        for attempt in range(self.config.max_generation_attempts):
            # Create grid with wall padding: (grid_size + 2*padding) x (grid_size + 2*padding)
            total_size = self.config.grid_size + 2 * self.config.wall_padding
            env = np.zeros((total_size, total_size), dtype=int)
            
            # Add wall borders
            env[0, :] = 1  # Top wall
            env[-1, :] = 1  # Bottom wall
            env[:, 0] = 1  # Left wall
            env[:, -1] = 1  # Right wall
            
            # Calculate inner navigable area (excluding walls)
            inner_start = self.config.wall_padding
            inner_end = self.config.grid_size + self.config.wall_padding
            
            # Place obstacles randomly in inner area only
            obstacle_density = random.uniform(*self.config.obstacle_density_range)
            inner_area_size = self.config.grid_size * self.config.grid_size
            num_obstacles = int(obstacle_density * inner_area_size)
            
            # Get inner area positions
            inner_positions = [(i, j) for i in range(inner_start, inner_end) 
                             for j in range(inner_start, inner_end)]
            random.shuffle(inner_positions)
            
            # Place obstacles in inner area
            for i in range(min(num_obstacles, len(inner_positions))):
                env[inner_positions[i]] = 1
            
            # Place start and goal in inner area
            start, goal = self._place_start_goal(env)
            
            # Validate environment
            if self._validate_environment(env, start, goal):
                return env, start, goal
        
        raise RuntimeError(f"Failed to generate valid environment after {self.config.max_generation_attempts} attempts")
    
    def _place_start_goal(self, env: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Place start and goal positions in inner navigable area"""
        # Only place start and goal in inner area (excluding walls)
        inner_start = self.config.wall_padding
        inner_end = self.config.grid_size + self.config.wall_padding
        
        empty_positions = [(i, j) for i in range(inner_start, inner_end) 
                          for j in range(inner_start, inner_end) if env[i, j] == 0]
        
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
    Extract robot's perception from environment + goal direction
    
    Biological Inspiration: Like how animals use peripheral vision combined 
    with goal-relative spatial awareness (compass-like navigation).
    
    State Representation: (local_view, goal_delta) = 11 features
    - local_view: 3×3 perception = 9 features
    - goal_delta: (dx, dy) = 2 features
    """
    
    def __init__(self, perception_size: int = 3, use_goal_delta: bool = True):
        """
        Initialize perception extractor with goal-aware navigation
        
        Args:
            perception_size: Size of perception window (3 for 3×3)
            use_goal_delta: If True, include goal relative coordinates
        """
        self.perception_size = perception_size
        self.use_goal_delta = use_goal_delta
    
    def extract_3x3_view(self, env: np.ndarray, robot_pos: Tuple[int, int]) -> np.ndarray:
        """
        Extract 3×3 perception view around robot position
        
        Args:
            env: Environment grid with wall padding
            robot_pos: Current robot position (x, y)
            
        Returns:
            3×3 binary grid: 0=free, 1=obstacle/wall
        """
        x, y = robot_pos
        
        # Extract 3×3 view centered on robot
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
    
    def calculate_goal_delta(self, robot_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate relative vector from robot to goal
        
        Args:
            robot_pos: Current robot position (x, y)
            goal_pos: Goal position (x, y)
            
        Returns:
            (dx, dy): Relative vector from robot to goal
        """
        dx = goal_pos[0] - robot_pos[0]  # x-direction
        dy = goal_pos[1] - robot_pos[1]  # y-direction
        return (dx, dy)
    
    def extract_goal_aware_perception(self, env: np.ndarray, robot_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> np.ndarray:
        """
        Extract complete state representation: (local_view, goal_delta)
        
        Args:
            env: Environment grid with wall padding
            robot_pos: Current robot position (x, y)
            goal_pos: Goal position (x, y)
            
        Returns:
            Combined feature vector: 9 perception + 2 goal_delta = 11 features
        """
        # Extract 3×3 perception view
        perception_view = self.extract_3x3_view(env, robot_pos)
        perception_features = perception_view.flatten()
        
        # Calculate goal delta
        goal_delta = self.calculate_goal_delta(robot_pos, goal_pos)
        goal_features = np.array([goal_delta[0], goal_delta[1]], dtype=np.float32)
        
        # Combine features
        if self.use_goal_delta:
            combined_features = np.concatenate([perception_features, goal_features])
        else:
            combined_features = perception_features
        
        return combined_features.astype(np.float32)
    
    
    def get_feature_count(self) -> int:
        """Get total number of features for current configuration"""
        perception_features = self.perception_size * self.perception_size
        goal_features = 2 if self.use_goal_delta else 0
        return perception_features + goal_features
    
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
        
        self.perception_extractor = PerceptionExtractor(
            perception_size=config.perception_size,
            use_goal_delta=config.use_goal_delta
        )
        
    def generate_complete_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate complete training dataset for robot navigation
        
        Returns:
        X_train: (n_examples, 11) - Robot perceptions with goal delta
        y_train: (n_examples,) - Optimal actions from A*
        metadata: List of environment metadata
        """
        
        all_perceptions = []
        all_actions = []
        all_metadata = []
        
        # Determine feature composition
        perception_features = self.config.perception_size * self.config.perception_size
        goal_features = 2 if self.config.use_goal_delta else 0
        total_features = perception_features + goal_features
        
        print(f"🧠 Generating training data for {self.config.num_environments} environments...")
        print(f"📊 Environment: {self.config.grid_size}×{self.config.grid_size} inner area with {self.config.wall_padding}-cell wall padding")
        print(f"📊 State representation: {perception_features} perception + {goal_features} goal_delta = {total_features} features")
        
        for env_idx in range(self.config.num_environments):
            if (env_idx + 1) % 100 == 0:
                print(f"   Progress: {env_idx + 1}/{self.config.num_environments} environments")
            
            try:
                # Step 1: Generate environment with wall padding
                env_with_walls, start, goal = self.env_generator.generate_environment()
                
                # Step 2: Find A* optimal path
                pathfinder = AStarPathfinder(env_with_walls)
                a_star_path = pathfinder.find_path(start, goal)
                
                if a_star_path is None:
                    continue
                
                # Step 3: Extract training examples from path
                env_perceptions = []
                env_actions = []
                
                for i in range(len(a_star_path) - 1):
                    current_pos = a_star_path[i]
                    next_pos = a_star_path[i + 1]
                    
                    # Convert movement to action
                    action = self.perception_extractor.movement_to_action(current_pos, next_pos)
                    
                    # Extract goal-aware perception: (local_view, goal_delta)
                    goal_aware_perception = self.perception_extractor.extract_goal_aware_perception(
                        env_with_walls, current_pos, goal
                    )
                    env_perceptions.append(goal_aware_perception)
                    env_actions.append(action)
                
                # Add to complete dataset
                all_perceptions.extend(env_perceptions)
                all_actions.extend(env_actions)
                
                # Store metadata
                # Count obstacles only in inner navigable area (excluding walls)
                inner_start = self.config.wall_padding
                inner_end = self.config.grid_size + self.config.wall_padding
                inner_area = env_with_walls[inner_start:inner_end, inner_start:inner_end]
                obstacle_count = np.sum(inner_area == 1)
                
                metadata = {
                    'env_idx': env_idx,
                    'start': start,
                    'goal': goal,
                    'path_length': len(a_star_path),
                    'obstacle_count': obstacle_count,
                    'obstacle_density': obstacle_count / (self.config.grid_size * self.config.grid_size),
                    'path': a_star_path,
                    'wall_padding': self.config.wall_padding
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
        
        # Feature breakdown
        feature_size = X_train.shape[1] if len(X_train.shape) > 1 else 1
        if feature_size == 11:
            print(f"Feature mode: Goal-aware (9 perception + 2 goal_delta = 11 features)")
        elif feature_size == 9:
            print(f"Feature mode: Basic (9 perception features only)")
        else:
            print(f"Feature mode: Unknown ({feature_size} features)")
        
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
    """Visualize a sample environment with path and wall padding"""
    
    # Create visualization
    vis_env = env.copy().astype(float)
    vis_env[start] = 0.5  # Start position
    vis_env[goal] = 0.8   # Goal position
    
    # Mark path
    for pos in path[1:-1]:  # Skip start and goal
        vis_env[pos] = 0.3
    
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_env, cmap='viridis')
    plt.title(f"{title} (with wall padding)")
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Add text annotations
    plt.text(start[1], start[0], 'R', ha='center', va='center', fontsize=16, color='white', weight='bold')
    plt.text(goal[1], goal[0], 'G', ha='center', va='center', fontsize=16, color='white', weight='bold')
    
    # Add grid to show inner navigable area
    inner_size = env.shape[0] - 2  # Assuming 1-cell wall padding
    for i in range(inner_size + 1):
        plt.axhline(y=0.5 + i, color='white', alpha=0.3, linewidth=0.5)
        plt.axvline(x=0.5 + i, color='white', alpha=0.3, linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Cell Type (0=free, 1=obstacle/wall)')
    
    plt.show()


def visualize_training_examples(X_train: np.ndarray, y_train: np.ndarray, num_examples: int = 9) -> None:
    """Visualize sample training examples with goal delta information"""
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(num_examples, len(X_train))):
        # Extract 3x3 perception and goal delta
        perception = X_train[i][:9].reshape(3, 3)  # First 9 features
        goal_delta = X_train[i][9:11]  # Last 2 features (dx, dy)
        action = y_train[i]
        
        axes[i].imshow(perception, cmap='gray', vmin=0, vmax=1)
        
        # Create title with goal delta info
        title = f'Action: {action_names[action]}\nGoal Δ: ({goal_delta[0]:.0f}, {goal_delta[1]:.0f})'
        axes[i].set_title(title, fontsize=10)
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
