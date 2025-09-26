# Training Data Analysis

## 🎯 Overview

This document explains the training data structure for the 2D Point-Robot Navigator project, where a robot learns to navigate using only a 3×3 perception window in 10×10 environments.

## 📊 Data Structure Summary

```
INPUT (X_train):                        OUTPUT (y_train):
┌─────────────────────────────────┐     ┌──────────────┐
│ Type: float32 (OPTIMIZED!)      │     │ Type: int8   │
│ Shape: (841, 9)                 │     │ Shape: (841,)│ For each sample there is corrsponding one action by Robot.
│ Values: 0.0 to 1.0              │     │ Values: 0-3  │
│ Meaning: 3×3 obstacle patterns  │     │ Meaning: Actions │
└─────────────────────────────────┘     └──────────────┘

X_train shape: (841, 9)
┌─────────────────────────────────┐
│ Sample 0: [0,1,0,0,0,1,0,1,0]   │  ← 9 features per sample 3×3 View along A* path
├─────────────────────────────────┤
│ Sample 1: [0,1,0,0,0,1,0,0,0]   │  ← 9 features per sample 3×3 View along A* path
├─────────────────────────────────┤
│ Sample 2: [1,0,0,1,0,0,0,0,1]   │  ← 9 features per sample 3×3 View along A* path
├─────────────────────────────────┤
│ ...                             │
├─────────────────────────────────┤
│ Sample 840: [0,0,1,0,1,0,0,0,0] │  ← 9 features per sample 3×3 View along A* path
└─────────────────────────────────┘

```

### 🔍 How to Visualize (841, 9) Shape:
Think of it as a Table
```
in_N → Input Neuron

| Sample | Feature1 (in_N1) | Feature2 (in_N2) | Feature3 (in_N3) | ... | Feature9 (in_N9) | Action          |
|--------|------------------|------------------|------------------|-----|------------------|-----------------|
|   0    |       0.0        |       1.0        |       0.0        | ... |       0.0        |        1        |
|   1    |       0.0        |       1.0        |       0.0        | ... |       0.0        |        1        |
|   2    |       1.0        |       0.0        |       0.0        | ... |       1.0        |        0        |
|  ...   |       ...        |       ...        |       ...        | ... |       ...        |       ...       |
|  840   |       0.0        |       0.0        |       1.0        | ... |       0.0        |        2        |

```

**Key Numbers:**
- **841 samples**: Total training examples
- **9 features**: Flattened 3×3 perception grid
- **4 actions**: UP(0), DOWN(1), LEFT(2), RIGHT(3)
- **100 environments**: Diverse training scenarios

## 🧠 How Training Data is Generated

### Step 1: Environment Creation
```
Generate 10×10 grid with random obstacles
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│ │█│ │ │█│ │ │ │ │ │  ← Random obstacles (█)
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │█│ │ │ │█│ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │█│ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │█│ │ │ │ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │█│ │ │ │ │█│ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │█│ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │█│ │ │ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │█│ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
Start: (0,0)  Goal: (9,9)
```

### Step 2: A* Pathfinding
```
A* algorithm finds optimal path
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│R│█│ │ │█│ │ │ │ │ │  ← R = Robot start
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│↓│ │█│ │ │ │█│ │ │ │  ← ↓ = Path direction
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│↓│ │ │ │ │ │ │█│ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│↓│█│ │ │ │ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│↓│ │ │█│ │ │ │ │█│ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│↓│ │ │ │ │█│ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│↓│ │█│ │ │ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│↓│ │ │ │ │ │█│ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│→→→→→→→→→G│  ← G = Goal
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
```

### Step 3: Extract 3×3 Perceptions
```
At each step along the A* path:

Position (0,0):                    Position (1,0):
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐            ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│R│█│ │ │█│ │ │ │ │ │            │ │█│ │ │█│ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤            ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │█│ │ │ │█│ │ │ │            │R│ │█│ │ │ │█│ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤            ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│ │█│ │ │ │ │ │█│ │ │            │ │ │ │ │ │ │ │█│ │ │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘            └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

3×3 View:                         3×3 View:
┌─────┬─────┬─────┐              ┌─────┬─────┬─────┐
│ 0.0 │ 1.0 │ 0.0 │  ← Out of    │ 0.0 │ 1.0 │ 0.0 │  ← Out of
├─────┼─────┼─────┤     bounds   ├─────┼─────┼─────┤     bounds
│ 0.0 │ 0.0 │ 1.0 │              │ 0.0 │ 0.0 │ 1.0 │
├─────┼─────┼─────┤              ├─────┼─────┼─────┤
│ 0.0 │ 1.0 │ 0.0 │              │ 0.0 │ 0.0 │ 0.0 │
└─────┴─────┴─────┘              └─────┴─────┴─────┘

Flattened: [0,1,0,0,0,1,0,1,0]   Flattened: [0,1,0,0,0,1,0,0,0]
Action: 1 (DOWN)                  Action: 1 (DOWN)
```

## 🔍 Data Structure Details

### Input Data (X_train)
- **Shape**: (841, 9)
- **Type**: float64
- **Values**: 0.0 (empty) to 1.0 (obstacle)
- **Structure**: Each row is a flattened 3×3 perception

### Output Data (y_train)
- **Shape**: (841,)
- **Type**: int64
- **Values**: 0-3 (action indices)
- **Mapping**: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

### Why These Data Types?

**Input (float32):**
- **Neural network compatibility**: Requires floating-point inputs
- **Obstacle probability**: Can represent partial visibility (0.0-1.0)
- **Standard practice**: Most ML libraries expect float64

**Why float32 anyway?**
- Neural Network Requirement: Most ML libraries expect float64 for inputs
- NumPy Default: np.array() creates float64 by default
- Future Flexibility: Could add partial visibility later (0.5 = partially visible obstacle)
- Library Compatibility: PyTorch/TensorFlow expect float inputs

**Output (int8):**
- **Discrete actions**: Only 4 possible values (0,1,2,3)
- **Memory efficient**: Integers use less memory than floats
- **Direct indexing**: Can be used as array indices

> **🤔 Why not int8 for 4 discrete actions?**
> 
> You're absolutely right! For only 4 values, we could use `int8`:
> ```python
> # Memory comparison for 841 samples:
> int64: 841 × 8 bytes = 6,728 bytes
> int8:  841 × 1 byte  = 841 bytes  # 8x less memory!
> ```
> 
> **Why we use int64 anyway:**
> - NumPy default behavior
> - ML library compatibility
> - Negligible impact for 841 samples (6KB vs 1KB)
> - Easy to optimize later: `y_train.astype(np.int8)`

## 🧬 Biological Inspiration

**Local Perception**: Like animals using limited peripheral vision to navigate
**Expert Demonstrations**: A* provides optimal "expert" decisions
**Pattern Learning**: Robot learns obstacle-action relationships through repetition

The robot learns to map 3×3 obstacle patterns to navigation actions, mimicking how animals use local sensory input to make movement decisions!

## 🔧 Neural Network Architecture

```
Input Layer: 9 neurons (3×3 flattened perception)
Hidden Layer 1: 64 neurons + ReLU + Dropout(0.2)
Hidden Layer 2: 32 neurons + ReLU + Dropout(0.2)
Output Layer: 4 neurons + Softmax
```

**Training Strategy:**
- Split: 80% train, 20% validation
- Batch size: 32-64
- Learning rate: 0.001 with decay
- Epochs: 50-100 with early stopping

## 📈 Key Insights

1. **Data Balance**: Action distribution is well-balanced (imbalance ratio: 1.25)
2. **Perception Complexity**: Average 2.05 obstacles per 3×3 view
3. **Environment Diversity**: 100 different environments with varying complexity
4. **Optimal Labels**: A* guarantees shortest path decisions
5. **Biological Connection**: Mimics animal navigation with limited vision

**Bottom Line:** The robot learns to navigate by observing optimal A* decisions at each position, building a comprehensive map of obstacle patterns to actions through supervised learning.