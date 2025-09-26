# 🧠 2D Point-Robot Navigator

A deep learning project that teaches a neural network to navigate 2D environments using limited 3x3 perception, inspired by biological navigation systems.

## 🎯 Project Overview

**Problem**: Learn a policy π that maps minimal robot sensing to actions that reach a goal without collisions in cluttered 2D worlds.

**Approach**: Supervised learning with expert demonstrations using A* pathfinding to generate optimal training data.

**Biological Inspiration**: Hippocampus place cells, visual cortex perception, and motor cortex learning.

## 🏗️ Project Structure

```
Proj_Robot_Nav/
├── core/                    # Core functionality
│   └── data_generation.py  # A* pathfinding, environment generation, training data
├── scripts/                 # Executable tools
│   └── generate_data.py     # Generate training datasets
├── tests/                   # Quality assurance
│   └── test_data_generation.py
├── configs/                 # Configuration files
│   ├── data_config.yaml     # Data generation parameters
│   └── model_config.yaml    # Neural network parameters
├── notebooks/               # Interactive analysis
│   └── 01_data_exploration.ipynb
├── data/                    # Data management
│   ├── raw/                 # Generated datasets
│   ├── processed/           # Preprocessed data
│   ├── models/              # Trained models
│   └── results/             # Experiment outputs
└── docs/                    # Documentation
    └── data_generation_pipeline.md
```

## 🚀 Quick Start

### Generate Training Data
```bash
python scripts/generate_data.py small    # 100 environments
python scripts/generate_data.py medium   # 500 environments  
python scripts/generate_data.py large    # 1000 environments
```

### Run Tests
```bash
python tests/test_data_generation.py
```

### Explore Data
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 📊 Data Pipeline

1. **Environment Generation**: 10x10 grid worlds with obstacles
2. **A* Pathfinding**: Optimal path generation using Manhattan distance
3. **Perception Extraction**: 3x3 robot vision around current position
4. **Action Encoding**: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
5. **Training Data**: (perception, action) pairs for supervised learning

## 🧠 Biological Connections

- **10x10 Environment**: Complete world (hidden from robot)
- **3x3 Perception**: Limited peripheral vision
- **A* Algorithm**: Expert navigation knowledge
- **Neural Network**: Motor cortex learning movement patterns
- **Training Process**: Learning from expert demonstrations

## 📈 Current Status

- ✅ **Data Generation**: Complete pipeline implemented
- ✅ **Environment Generation**: A* pathfinding with validation
- ✅ **Training Data**: 3x3 perceptions + optimal actions
- 🔄 **Neural Network**: Next phase - model architecture and training
- 🔄 **Evaluation**: Future - model performance analysis

## 🛠️ Development

This project follows modular architecture with clear separation of concerns:
- **Core modules** are reusable and independent
- **Scripts** orchestrate core functionality
- **Tests** verify system behavior
- **Configs** manage parameters
- **Data** is organized by pipeline stage

## 📚 Documentation

- [Data Generation Pipeline](docs/data_generation_pipeline.md) - Complete technical details
- [Data Directory](data/README.md) - Data management guide
- [Project Structure](.cursor/rules/project-structure.mdc) - Architecture guidelines
