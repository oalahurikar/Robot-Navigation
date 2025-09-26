## Project: 2D Point-Robot Navigator
Problem statement: Learn a policy π that maps minimal robot sensing to actions that reach a goal without collisions in cluttered 2D worlds.

## Complete Data Generation Pipeline. Step-by-Step Process:
```
1. Generate 10x10 Environment
   ↓
2. Use A* to find optimal path
   ↓
3. Extract 3x3 perceptions along path
   ↓
4. Convert movements to 4 actions
   ↓
5. Create training examples
```

## Data Generation & Tagging Pipeline for NN training.
### Phase 1: Core Functionality

Step 1: Environment Generation (World)
Objective: Create diverse, random 2D grid worlds for training
Minimal Requirements:
- Grid Size: 10x10 (start simple, scale up later)
- Obstacle Density: 15-25% coverage (adjustable parameter)
- Start/Goal Placement: Random valid positions
- Validation: Ensure path exists between start and goal

🗺️ ENVIRONMENT:
"R = Robot, G = Goal, X = Obstacle, . = Empty"
R . X . . 
. X X . . 
. . . . . 
X . . X . 
. . . . G 

Key Design Decisions:
```
1. Obstacle Patterns:
   - Random scattered obstacles

2. Difficulty Levels:
   - Easy: Sparse obstacles, direct paths
   - Medium: Moderate obstacles, some detours needed
   - Hard: Dense obstacles, complex navigation required

3. Environment Types:
   - Maze-like structures
   - Open Spaces with Scattered Obstacles
   ```
Example A - Simple Maze:
R X X X X X X X X X
. . . . . . . . . X
X X X X X X X X . X
X . . . . . . . . X
X . X X X X X X X X
X . X . . . . . . .
X . X . X X X X X .
X . . . X . . . . .
X X X X X . X X X G
. . . . . . . . . .

Example B - Complex Maze:
R . X X X X X X X X
X . . . . . . . . X
X X X . X X X X . X
. . . . X . . . . X
. X X X X . X X X X
. . . . . . X . . .
X X X . X X X . X .
X . . . X . . . X .
X . X X X . X X X G
X . . . . . . . . .

Example A - Sparse (15% density):
R . . . . . . . . G
. . X . . . . X . .
. . . . . . . . . .
. X . . . . . . X .
. . . . . . . . . .
. . . . . . . . . .
. . X . . . . . . .
. . . . . . . . . .
. X . . . . . . . .
. . . . . . . . . .

Example B - Dense (25% density):
R . X . . . . X . G
. X . X . . X . X .
. . . . . . . . . .
. . X . . X . . . .
. . . . . . . . . .
X . . . . . . . X .
. . X . . . . . . .
. . . . X . . . . .
. X . . . . . X . .
. . . . . . . . . .

---

Step 2: Expert Pathfinding (A* Algorithm)
Objective: Generate optimal paths for each environment
A* Implementation Strategy:
1. Use Manhattan distance as heuristic
2. Handle 4-connected movement (up, down, left, right)
3. Return complete path from start to goal
4. Validate path exists before proceeding
Path Quality Assurance:
- Check path exists (no impossible scenarios)
- Verify path is optimal (shortest possible)
- Handle edge cases (start = goal, blocked paths)

---
Step 3: State Extraction   
Objective: Extract robot's perception at each step
- Robot sees 3x3 grid around its position
- Input size: 9 values (flattened 3x3)
- Biological inspiration: Limited peripheral vision


---
