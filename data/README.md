# Data Directory Structure

This directory contains all data files for the 2D Point-Robot Navigator project.

## Directory Structure

```
data/
├── raw/                    # Original generated datasets
│   ├── small_training_dataset.npz
│   ├── medium_training_dataset.npz
│   └── large_training_dataset.npz
├── processed/              # Preprocessed data ready for training
│   ├── train_data.npz
│   ├── validation_data.npz
│   └── test_data.npz
├── models/                 # Trained neural network models
│   ├── checkpoints/        # Training checkpoints
│   └── final_models/       # Final trained models
└── results/                # Experiment results and outputs
    ├── logs/               # Training logs
    ├── visualizations/     # Plots and charts
    └── metrics/            # Performance metrics
```

## Data Pipeline

1. **Raw Data Generation**: `scripts/generate_data.py` → `data/raw/`
2. **Data Preprocessing**: Raw data → `data/processed/`
3. **Model Training**: Processed data → `data/models/`
4. **Results Storage**: Training outputs → `data/results/`

## File Naming Conventions

- **Raw datasets**: `{size}_training_dataset.npz` (e.g., `small_training_dataset.npz`)
- **Processed data**: `{split}_data.npz` (e.g., `train_data.npz`)
- **Models**: `model_{timestamp}.pth` (e.g., `model_20241201_143022.pth`)
- **Results**: `experiment_{name}_{timestamp}` (e.g., `experiment_baseline_20241201`)

## Usage

### Generate Raw Data
```bash
python scripts/generate_data.py small    # → data/raw/small_training_dataset.npz
python scripts/generate_data.py medium   # → data/raw/medium_training_dataset.npz
python scripts/generate_data.py large    # → data/raw/large_training_dataset.npz
```

### Load Data in Code
```python
from core.data_generation import load_training_data

# Load raw data
X_train, y_train, metadata = load_training_data("data/raw/small_training_dataset.npz")

# Load processed data
X_train, y_train, metadata = load_training_data("data/processed/train_data.npz")
```

## Notes

- Large data files are excluded from version control (see `.gitignore`)
- Keep raw data files for reproducibility
- Processed data can be regenerated from raw data
- Models and results should be backed up separately
