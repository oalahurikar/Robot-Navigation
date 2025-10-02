# 📊 Distance-Based Perception Notebook Guide

## ✅ What Was Updated

The `01_data_exploration.ipynb` notebook has been updated to fully support **distance-based perception** visualization and analysis.

---

## 🔄 Key Changes

### 1. **Cell 6: Data Loading**
- Now loads `distance_5x5_large.npz` dataset
- **Auto-detects** perception type (binary vs distance-based)
- Displays perception type in output

### 2. **Cell 10: Sample Data Display**
- Shows **continuous distance values** (0.0 to 1.0)
- Instead of binary symbols (X and .)
- Format: `0.0 0.2 0.4 0.6 0.8` instead of `X . . . .`

### 3. **Cell 13: Perception Statistics**
- **Distance-based mode** shows:
  - Mean/std/min/max distance values
  - Immediate obstacle counts (distance=0.0)
  - Distance distribution bins (0.0-0.2, 0.2-0.4, etc.)
  - Closest obstacle statistics
- **Binary mode** shows:
  - Traditional obstacle counts

### 4. **NEW Cell 14: Distance Field Visualizations** 🎨
Added comprehensive distance-specific visualizations:
- **Heatmaps** (6 sample distance fields with color coding)
  - Green = far from obstacles (safe)
  - Yellow = medium distance
  - Red = close to obstacles (dangerous)
- **Distance distribution histogram**
- **Immediate obstacles per view**
- **Mean distance by action**
- **Closest obstacle distribution**
- **Distance field variance**
- **Average distance field heatmap** (across all data)

### 5. **Cell 15: Updated Visualizations**
- Third subplot adapts to perception type
- Shows immediate obstacles for distance-based
- Shows all obstacles for binary

### 6. **Cell 16: Updated Summary**
- Highlights distance-based perception benefits
- Lists expected performance improvements

---

## 🎯 How to Use the Updated Notebook

### Step 1: Open Notebook
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Step 2: Run All Cells
- **Kernel → Restart & Run All**
- Or run cells sequentially with Shift+Enter

### Step 3: Explore Visualizations

#### What You'll See:

**Binary Perception (Old)**:
```
3×3 Perception (Binary):
. X .
. . .
X . .
```

**Distance Perception (New)** 🎯:
```
5×5 Perception (Distance):
0.4 0.0 0.2 0.4 0.6
0.6 0.2 0.4 0.6 0.8
0.8 0.4 0.6 0.8 1.0
0.6 0.2 0.4 0.6 0.8
0.4 0.0 0.2 0.4 0.6
```

---

## 📊 Understanding Distance Values

| Value | Meaning | Color (Heatmap) | Interpretation |
|-------|---------|-----------------|----------------|
| **0.0** | Immediate obstacle | 🔴 Red | Dangerous - obstacle here |
| **0.2** | Distance = 1 cell | 🟠 Orange | Very close |
| **0.4** | Distance = 2 cells | 🟡 Yellow | Close |
| **0.6** | Distance = 3 cells | 🟢 Light Green | Medium |
| **0.8** | Distance = 4 cells | 🟢 Green | Far |
| **1.0** | Distance = 5+ cells | 🟢 Dark Green | Very far (safe) |

### Formula:
```python
normalized_distance = min(actual_distance / max_sensing_distance, 1.0)
# With max_sensing_distance = 5:
# actual_distance = 1 → 0.2
# actual_distance = 2 → 0.4
# actual_distance = 3 → 0.6
# actual_distance = 5+ → 1.0
```

---

## 🎨 New Visualizations Explained

### 1. Distance Field Heatmaps (Top 6 panels)
- **Shows**: Individual perception examples as colored grids
- **Purpose**: See spatial distance gradients
- **Insight**: Robot can "feel" how far obstacles are in each direction

### 2. Distance Value Distribution
- **Shows**: Histogram of all distance values
- **Purpose**: Understand data distribution
- **Expected**: Should have mix of all values (0.0 to 1.0)

### 3. Immediate Obstacles per View
- **Shows**: How many cells have distance=0.0 (obstacles)
- **Purpose**: Compare with binary "obstacle count"
- **Insight**: Most views have 2-4 immediate obstacles

### 4. Mean Distance by Action
- **Shows**: Average distance for each action direction
- **Purpose**: See if certain actions correlate with safety
- **Insight**: All actions should have similar mean distances

### 5. Closest Obstacle Distribution
- **Shows**: Minimum distance in each perception view
- **Purpose**: Safety margin analysis
- **Insight**: How close does robot get to obstacles?

### 6. Distance Field Variance
- **Shows**: How varied distances are within each view
- **Purpose**: Complexity measure
- **Insight**: High variance = complex environment

### 7. Average Distance Field
- **Shows**: Mean distance at each grid position (across all data)
- **Purpose**: Identify common patterns
- **Insight**: Center should have higher values (robot position is usually clear)

---

## 🔬 Comparing Binary vs Distance-Based

| Aspect | Binary | Distance-Based |
|--------|--------|----------------|
| **Values** | 0.0 or 1.0 only | Continuous 0.0-1.0 |
| **Information** | Presence/absence | Proximity measure |
| **Visualization** | X and . symbols | Numeric values |
| **Heatmaps** | Black/white | Color gradient |
| **Statistics** | Obstacle counts | Distance distributions |
| **Neural Network** | Learns patterns | Learns gradients |
| **Generalization** | Limited (~50% novel) | Better (~75-80% novel) |
| **Real-World** | Simulated only | LIDAR-compatible |

---

## 🚀 Next Steps After Visualization

1. **Train Neural Network**:
   ```bash
   python scripts/train_nn.py --perception 5x5
   ```

2. **Compare with Binary Baseline**:
   - Train with `data/raw/binary_5x5_small.npz`
   - Train with `data/raw/distance_5x5_large.npz`
   - Compare validation accuracy

3. **Test on Novel Environments**:
   - Generate new test environments
   - Evaluate generalization performance

4. **Analyze Learned Behaviors**:
   - Visualize activation patterns
   - Study failure cases
   - Identify emergent behaviors

---

## 🐛 Troubleshooting

### Issue: Notebook shows binary visualization
**Solution**: Make sure cell 6 loads `distance_5x5_large.npz` not `large_training_dataset.npz`

### Issue: Heatmaps not showing
**Solution**: Run cell 14 (distance visualizations cell) - it only runs for distance-based data

### Issue: Old data showing
**Solution**: 
```python
# In cell 6, ensure you have:
X_train, y_train, metadata = load_training_data("../data/raw/distance_5x5_large.npz")
```

### Issue: "is_distance_based is not defined"
**Solution**: Run cells in order, starting from cell 1. Cell 6 defines this variable.

---

## 📖 Related Documentation

- **Implementation Details**: `docs/distance_based_implementation_log.md`
- **Solution Overview**: `docs/solution_3_distance_based_perception.md`
- **Configuration**: `configs/data_config.yaml`
- **Data Generation**: `scripts/generate_data.py --help`

---

## ✨ Key Takeaways

1. **Distance-based perception provides continuous spatial information**
2. **Heatmap visualizations reveal distance gradients**
3. **Neural network learns from richer data**
4. **Better generalization expected (theory: +25-30%)**
5. **Direct transfer to real robots (LIDAR mapping)**

---

**Status**: ✅ Notebook fully updated and ready for distance-based data exploration!

**Author**: Robot Navigation Project  
**Date**: September 30, 2025  
**Branch**: `Distance-Based-Perception-System`

