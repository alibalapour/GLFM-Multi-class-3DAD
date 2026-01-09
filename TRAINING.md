# GLFM Model Training Guide

## Overview

This document explains how the GLFM (Global-Local Feature Matching) model is trained for multi-class 3D point cloud anomaly detection. The model is designed to work with MVTec 3D-AD and Real3D-AD datasets and uses a three-stage training approach.

## Three-Stage Architecture

The GLFM method consists of three main stages:

### Stage I: Anomaly Synthesis and Self-Supervised Training
Self-supervised pre-training of the point cloud feature extractor using synthetic anomalies.

### Stage II: Memory Bank Establishment
Building global and local memory banks from normal training data to capture feature distributions.

### Stage III: Anomaly Detection
Testing phase where anomalies are detected by measuring feature distances from the memory banks.

---

## Real3D-AD Dataset Preprocessing

### Dataset Structure

The Real3D-AD dataset consists of 12 object categories:
- airplane, candybar, car, chicken, diamond, duck
- fish, gemstone, seahorse, shell, starfish, toffees

**Dataset Location:** `/data/Real3d_cut`

### Preprocessing Steps for Real3D-AD

The Real3D-AD dataset undergoes specific preprocessing that differs from MVTec 3D-AD. Here's the detailed process:

#### 1. **Training Data Preprocessing** (`data/real3d.py` - `Real3DTrain` class)

**File Format:**
- Training data is stored as `.asc` files (ASCII point cloud format)
- Located in: `{DATASET_PATH}/{class_name}/train_cut/`

**Loading Process:**
```python
# Load point cloud from .asc file
input_points = np.loadtxt(pcd_path, dtype=np.float32)

# Extract XYZ coordinates (first 3 columns)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(input_points[:, 0:3])
```

**Voxel Downsampling:**
```python
# Downsample to reduce point density and normalize point count
voxel_size_setting = 0.15  # Can be adjusted (default: 0.15, alternative: 0.2)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
```

This downsampling step:
- Reduces computational requirements
- Normalizes point cloud density across different scans
- Maintains spatial structure while reducing redundancy

**Key Difference from MVTec 3D-AD:**
- Real3D-AD uses raw point clouds (`.asc` format) instead of organized TIFF images
- No RGB information is processed for Real3D-AD
- Training data comes from a `train_cut` directory (pre-cut/cropped data)

#### 2. **Testing Data Preprocessing** (`data/real3d.py` - `Real3DTest` class)

**File Format:**
- Test data is stored as `.txt` files
- Located in: `{DATASET_PATH}/{class_name}/test/{defect_type}/`

**Special Two-Part Point Cloud Structure:**

Real3D-AD test data contains points with labels (4th column):
- Column 4 = 0: Normal/reference points
- Column 4 = 1: Anomalous points

**Dual Point Cloud Processing:**
```python
# Separate points based on label
idx1 = input_points[:, 3] == 0  # Normal points
idx2 = input_points[:, 3] == 1  # Anomalous points

# Create two separate point clouds
pcd1.points = o3d.utility.Vector3dVector(input_points[idx1, 0:3])
pcd2.points = o3d.utility.Vector3dVector(input_points[idx2, 0:3])

# Combine for downsampling
pcd = pcd1 + pcd2
pcd_new = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
```

**Maintaining Point Correspondence After Downsampling:**

After voxel downsampling, the preprocessing re-associates each downsampled point with its original label:

```python
# Use KD-Tree to find which original point each downsampled point corresponds to
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
for x in pcd_points_new:
    [k, idx_, _] = pcd_tree.search_knn_vector_3d(x, 1)
    if idx_[0] < pc_len:  # pc_len = len(pcd1.points)
        pcd1_vec.append(x)  # Normal point
    else:
        pcd2_vec.append(x)  # Anomalous point
```

**Ground Truth Generation:**
```python
# Create binary ground truth labels
gt = torch.zeros(unorganized_pc.shape[0])
gt[len(pcd1.points):len(pcd.points)] = 1.0  # Mark anomalous points

# Convert to binary mask
gt = torch.where(gt > 0.5, 1., 0.)
```

**Key Differences from MVTec 3D-AD:**
- Real3D-AD test files contain both normal and anomalous points in the same file
- No separate GT images; ground truth is derived from point labels
- More complex preprocessing to maintain label correspondence after downsampling

#### 3. **No Background Removal**

Unlike MVTec 3D-AD (which requires plane/background removal), Real3D-AD:
- Comes pre-processed with training data already cut/cropped
- Does not require plane segmentation or background removal
- Does not use the `preprocessing.py` script (which is MVTec 3D-AD specific)

---

## Stage I: Self-Supervised Training with Anomaly Synthesis

### Purpose
Adapt the point cloud feature extractor to better distinguish between normal and anomalous features by training on synthetic anomalies.

### Synthetic Anomaly Generation (`self_supervised/generateAD.py`)

The **stretching** method creates synthetic anomalies by deforming local regions:

**Algorithm:**
1. **Normal Estimation:** Compute surface normals for all points using hybrid search
2. **Region Selection:** 
   - Randomly select 1-2% of points as anomaly centers
   - Choose stretch direction (upward/downward along normal)
3. **Point Cloud Scaling:** Apply random scaling (80-120%) in x, y, z directions
4. **Neighborhood Deformation:**
   - Find k-nearest neighbors (k = selected_point_num) in scaled space
   - Stretch points along the surface normal
   - Deformation intensity decreases with distance from center

**Parameters:**
- `selected_point_num`: Random between `tot_points/100` and `tot_points/50`
- `dis_points`: Average distance between neighboring points
- Stretch magnitude: `(k-i) * dis_points * 100 * direction`

**Output:**
- Deformed point cloud with anomalous regions
- Binary mask: 1 for deformed points, 0 for normal points

### Training Process (`train.py`)

**Model:** PointTransformer with PointMAE backbone
- **Group size:** 128 points
- **Number of groups:** 1024
- **Feature extraction layer:** [11]

**Training Configuration:**
```python
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Loss Functions
- FocalLoss: Handles class imbalance
- BinaryDiceLoss: Optimizes for segmentation overlap

# Learning Rate Schedule (Note: Code has very aggressive decay)
- Epoch 0-39: lr = 0.0001
- Epoch 40-79: lr *= 0.00001 (becomes 1e-9)
- Epoch 80-99: lr *= 0.000001 (becomes 1e-15)

Note: The learning rate schedule in train.py appears to have extremely aggressive 
decay that may not provide meaningful updates in later epochs. This may be a 
bug in the original code (uses undefined 'optimizer_Adam' instead of 'optimizer').
```

**Training Loop:**
1. Generate synthetic anomalies on-the-fly (p=1.0, always generate)
2. Extract features using PointTransformer
3. Interpolate features to original point resolution
4. Classify each point as normal/anomalous
5. Compute combined loss: `FocalLoss + DiceLoss`
6. Update model weights

**Iterations per Epoch:** 100 (limited for efficiency)

**Checkpoint:** Save model weights after each epoch to `./weights/point_transformer_epoch_{epoch}.pth`

**Final Model:** `pointmae_adapted.pth` - Used for Stage II and III

---

## Stage II: Memory Bank Establishment

### Training Phase (`patchcore_runner.py` - `fit` method)

**Process:**
1. **Feature Extraction:**
   ```python
   # For each training sample
   for pc, _, _, path in train_loader:
       method.collect_features(pc)  # Extract global and local features
   ```

2. **Feature Collection** (`feature_extractors/GLFM.py`):
   - **Point-level Features:** Extract using adapted PointTransformer
   - **Local Features:** 
     - Use FPS (Farthest Point Sampling) to select 1024 centers
     - Use KNN (k=128) to group points
     - Average features within each group
   - **Global Features:** Average all local features

3. **Clustering** (`method.divide_bank()`):
   - Cluster training samples based on global features
   - Number of clusters: `k_class` (e.g., 1 for single-class, 10 for multi-class MVTec, 3 for Real3D)

4. **Memory Bank Creation** (`method.add_sample_to_mem_bank()`):
   - **Global Memory Bank:** Store representative global features from each cluster
   - **Local Memory Bank:** Store local patch features from training samples

5. **Coreset Selection** (`method.run_coreset()`):
   - Select representative subset of features
   - Reduces memory and computational requirements during inference

**Real3D-AD Specific Settings:**
- **Single-Class:** `k_class=1` (one memory bank per object category)
- **Multi-Class:** `k_class=3` (three clusters across all categories)

---

## Stage III: Anomaly Detection

### Testing Phase (`patchcore_runner.py` - `evaluate` method)

**Process:**
1. **Feature Extraction:**
   ```python
   for pc, mask, label, path in test_loader:
       method.predict(pc, mask, label, path)
   ```

2. **Anomaly Scoring:**
   - Extract global and local features from test sample
   - **Global Matching:** Compute distance to nearest neighbor in global memory bank
   - **Local Matching:** Compute distance to nearest neighbor in local memory bank
   - **Combined Score:** Aggregate global and local anomaly scores

3. **Metrics Calculation:**
   - **Image-level AUROC:** Classify entire point cloud as normal/anomalous
   - **Pixel-level AUROC:** Classify each point as normal/anomalous
   - **AU-PRO:** Area Under Per-Region-Overlap curve

**Real3D-AD Ground Truth:**
- For "good" samples: All points labeled as normal (gt=0)
- For defective samples: Points separated into normal (pcd1) and anomalous (pcd2) based on 4th column in input file

---

## Complete Training Workflow

### Prerequisites

1. **Environment Setup:**
   ```bash
   conda create --name GLFM python=3.8
   conda activate GLFM
   conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
   pip install tifffile open3d-cpu
   pip install -r requirements.txt
   pip install --upgrade KNN_CUDA-0.2-py3-none-any.whl
   pip install Pointnet2_PyTorch
   ```

2. **Dataset Preparation:**
   - Download Real3D-AD dataset (preprocessed version recommended)
   - Place in `/data/Real3d_cut` or update `DATASETS_PATH` in `data/real3d.py`
   - Dataset structure:
     ```
     Real3d_cut/
     ├── airplane/
     │   ├── train_cut/*.asc
     │   └── test/
     │       ├── good/*.txt
     │       └── {defect_type}/*.txt
     ├── candybar/
     ...
     ```

### Training Commands

#### Stage I: Self-Supervised Training
```bash
python train.py
```
- Input: MVTec 3D-AD training data (uses MVTec for pretraining)
- Output: `./weights/point_transformer_epoch_*.pth`
- Rename best checkpoint to: `pointmae_adapted.pth`

#### Stage II & III: Single-Class Anomaly Detection
```bash
python main.py --dataset real --task Single-Class --k_class 1 --model_pth ./pointmae_adapted.pth
```

#### Stage II & III: Multi-Class Anomaly Detection  
```bash
python main.py --dataset real --task Multi-Class --k_class 3 --model_pth ./pointmae_adapted.pth
```

**Arguments:**
- `--dataset`: 'real' for Real3D-AD, 'mvtec' for MVTec 3D-AD
- `--task`: 'Single-Class' or 'Multi-Class'
- `--k_class`: Number of clusters (1 for single-class, 3 for Real3D multi-class, 10 for MVTec multi-class)
- `--model_pth`: Path to adapted PointMAE model
- `--fetch_idx`: Feature extraction layers (default: [0,1])

---

## Key Code Files Reference

### Real3D-AD Specific Files:
- **`data/real3d.py`**: Real3D-AD dataset loader and preprocessing
  - `Real3DTrain`: Training data loader with voxel downsampling
  - `Real3DTest`: Test data loader with dual point cloud handling

### Model Training:
- **`train.py`**: Stage I self-supervised training script
- **`self_supervised/generateAD.py`**: Synthetic anomaly generation (stretching method)
- **`self_supervised/dataset.py`**: Dataset class for self-supervised training

### Feature Extraction and Detection:
- **`feature_extractors/GLFM.py`**: GLFM feature extraction implementation
- **`feature_extractors/models.py`**: PointTransformer architecture
- **`patchcore_runner.py`**: Main training and evaluation orchestrator

### Main Entry Points:
- **`main.py`**: Entry point for Stage II & III (memory bank + detection)
- **`run.sh`**: Example commands for different training scenarios

---

## Important Notes for Real3D-AD

1. **No RGB Information:** Real3D-AD only uses XYZ coordinates, unlike MVTec 3D-AD which uses RGB+XYZ

2. **Pre-cut Data:** Training data is already cropped/cut (train_cut directory), no additional preprocessing needed

3. **Voxel Size:** Critical parameter for balancing point count and spatial resolution
   - Default: 0.15
   - Alternative: 0.2 (fewer points, faster processing)
   - Adjust in `data/real3d.py` (line 28)

4. **Ground Truth Format:** Test files contain labels in 4th column (0=normal, 1=anomalous)

5. **Memory Requirements:** Real3D-AD test files can be large; voxel downsampling is essential

6. **Performance Note:** Paper reports better results when Real3D-AD is converted to TIFF format with organized structure (similar to MVTec 3D-AD), but current implementation uses raw point clouds.

---

## Expected Results

### Real3D-AD Multi-Class (k_class=3):
| Metric | Mean Performance |
|--------|------------------|
| Image ROCAUC | 0.611 |
| Pixel ROCAUC | 0.649 |

### Real3D-AD Single-Class (k_class=1):
| Metric | Mean Performance |
|--------|------------------|
| Image ROCAUC | 0.679 |
| Pixel ROCAUC | 0.684 |

Note: Results may vary from paper due to using raw point clouds instead of TIFF format.

---

## Troubleshooting

**Issue:** Out of memory during training
- **Solution:** Reduce `num_group` in PointTransformer or increase `voxel_size_setting`

**Issue:** Poor performance on Real3D-AD
- **Solution:** Check voxel size, ensure proper data format, consider converting to TIFF

**Issue:** KNN_CUDA installation fails
- **Solution:** Download pre-built wheel from GitHub releases

**Issue:** Dataset path not found
- **Solution:** Update `DATASETS_PATH` in `data/real3d.py` to match your local setup

---

## References

- Paper: "Boosting Global-Local Feature Matching via Anomaly Synthesis for Multi-Class Point Cloud Anomaly Detection" (IEEE TASE 2025)
- Based on: [3D-ADS](https://github.com/eliahuhorwitz/3D-ADS) and [M3DM](https://github.com/nomewang/M3DM)
