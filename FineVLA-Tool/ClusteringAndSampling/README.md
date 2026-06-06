# VLA Trajectory Similarity Analysis and Clustering

Uses **DTW (Dynamic Time Warping)** to perform **temporal elastic alignment** on EEF pose trajectories in VLA datasets, ignoring differences in execution speed and focusing only on spatial shape and pose sequences, thereby analyzing similarity between different trajectories and performing clustering.

## Core Idea

### Why can't we compare frames directly?

The same action can be performed faster or slower, and even with **locally inconsistent speeds** (one segment of trajectory A is faster than B, while another segment is slower). Direct frame-by-frame comparison yields incorrect distances due to temporal misalignment.

### How does DTW solve this?

DTW finds a **nonlinear alignment path** between two trajectories:
- In some segments, multiple frames of A align to one frame of B (A is slower in that segment)
- In other segments, one frame of A aligns to multiple frames of B (A is faster in that segment)
- In other segments, frames align one-to-one

The alignment is **segment-by-segment, position-by-position**, rather than using a single scaling factor for the entire trajectory, thus handling locally inconsistent speeds.

## DTW Algorithm in Detail

### 1. Problem Definition

Given two trajectories:
- Trajectory A: `a_1, a_2, ..., a_N` (N frames)
- Trajectory B: `b_1, b_2, ..., b_M` (M frames)

Each frame is an 8-dimensional vector `[x, y, z, qx, qy, qz, qw, gripper]`.

The goal of DTW is to find a **warp path** `W = (w_1, w_2, ..., w_K)`, where each `w_t = (i_t, j_t)` means "frame i_t of trajectory A aligns with frame j_t of trajectory B", minimizing the total cost along the path.

### 2. Per-frame Cost Function

The cost for each frame pair `(a_i, b_j)` is a weighted combination of three components:

| Component | Formula | Description |
|-----------|---------|-------------|
| **Position** | `d_pos = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)` | Euclidean distance of EEF position (meters) |
| **Orientation** | `d_rot = 2*arccos(min(\|q1*q2\|, 1.0))` | Quaternion geodesic angular distance (radians) |
| **Gripper** | `d_grip = \|g1 - g2\|` | Gripper openness difference |

Total cost:

```
cost(i, j) = w_pos * d_pos + w_rot * d_rot + w_grip * d_grip
```

**About quaternion geodesic angular distance**:

A unit quaternion `q = (qx, qy, qz, qw)` represents a 3D rotation. The rotation angle between two orientations equals the angle of the relative rotation quaternion, and the scalar part of the relative rotation quaternion is precisely the 4D dot product of the two quaternions. Therefore:

```
d_rot = 2 * arccos(|q1 * q2|)
```

- The absolute value `|q1*q2|` is used because `q` and `-q` represent the same rotation
- Clipping to 1.0 prevents `arccos` from returning NaN due to floating-point errors
- The range is `[0, pi]`, i.e., 0 to 180 degrees

### 3. Dynamic Programming Recurrence

DTW uses dynamic programming to find the optimal alignment path. Define `dp[i][j]` as the optimal cumulative cost for aligning the first i frames of trajectory A with the first j frames of trajectory B.

**Recurrence formula**:

```
dp[0][0] = 0
dp[i][j] = cost(i, j) + min(dp[i-1][j-1],   <- diagonal: one-to-one match
                             dp[i-1][j],       <- vertical: current frame of A "repeats" to align multiple frames of B
                             dp[i][j-1])        <- horizontal: current frame of B "repeats" to align multiple frames of A
```

**Intuitive understanding** -- the three transition directions correspond to three temporal alignment modes:

```
           j (Trajectory B)
           ->
    +------------------+
    |  \  <- diagonal(1:1)  |
  i |   \              |
(A) |    \             |
  v | v repeat A frame |
    |     -> repeat B frame |
    +------------------+
```

- **Diagonal `dp[i-1][j-1]`**: Frame i of A matches frame j of B one-to-one, both sides advance in sync
- **Vertical `dp[i-1][j]`**: A advances one frame, B stays -- A is slower in this segment (multiple A frames align to the same B frame)
- **Horizontal `dp[i][j-1]`**: B advances one frame, A stays -- B is slower in this segment (multiple B frames align to the same A frame)

The final `dp[N][M]` is the raw DTW distance between the two trajectories.

**Time complexity**: O(N * M). For trajectories of ~700 frames, the matrix size is about 700x700 = 490,000 elements.

### 4. Sakoe-Chiba Window Constraint

Unconstrained DTW allows arbitrarily warped alignment paths, but:
- Computation is the full O(N * M)
- Pathological alignments may occur (e.g., frame 1 of A matching the last frame of B)

The **Sakoe-Chiba window** restricts the alignment path to a band of +/-w around the diagonal:

```
           j (Trajectory B)
    +------------------+
    |\###              |   # = allowed computation region
    | \####            |   (diagonal +/- window)
    |  \####           |
    |   \####          |   blank = skipped, stays at infinity
    |    \####         |
    |     \####        |
    |      \###        |
    +------------------+
```

- Only `|i - j| <= w` is computed on the alignment path; `dp[i][j]` beyond the range stays at infinity
- Complexity reduces to **O(N * window)**, e.g., with window=100, only 700x200 = 140,000 elements need computing
- When `|N - M| > window`, the window automatically expands to `max(window, |N - M|)` to ensure path reachability

### 5. Path Backtracing and Normalization

The raw DTW distance `dp[N][M]` grows with trajectory length (longer paths accumulate more cost). To make distances comparable between trajectory pairs of different lengths, **normalization** is needed.

**Backtracing**: Starting from `dp[N][M]`, at each step choose the smallest predecessor among `dp[i-1][j-1]`, `dp[i-1][j]`, `dp[i][j-1]`, until reaching `dp[0][0]`. The number of steps traversed is the path length `L`.

```
Normalized distance = dp[N][M] / L
```

This gives the "average cost per step" for each trajectory pair, eliminating the amplification effect of trajectory length on distance.

**Windowed mode** normalization: Since the full dp matrix is not saved (only the final value is returned), approximate normalization `dp[N][M] / (N + M)` is used, where `N + M` is an upper bound on path length.

### 6. Complete Computation Flow (Two Trajectories as Example)

```
Trajectory A: (649, 8)   Trajectory B: (771, 8)
         |                    |
         v                    v
    +-------------------------------+
    |  For each (i,j) compute       |  -> position L2 + quaternion angle + gripper diff
    |  frame_cost                   |
    +-------------------------------+
                  |
                  v
    +-------------------------------+
    |  DP table fill: dp[i][j] =    |
    |    cost(i,j) + min(3 predecessors) |  -> O(649 * 771) ~ 500K iterations
    +-------------------------------+
                  |
                  v
    +-------------------------------+
    |  dp[649][771] = raw DTW dist  |
    |  Backtrace path length L~1000 |
    |  Normalized dist = dp / L     |
    +-------------------------------+
                  |
                  v
           Distance ~ 0.329
```

### 7. Batch Distance Matrix

For N trajectories, `N * (N-1) / 2` pairs need to be computed (symmetric matrix, only upper triangle).

- 100 trajectories -> 4950 pairs
- Supports `n_jobs > 1` for parallel acceleration via `multiprocessing.Pool`
- After computation, automatically cached as `.npz` files; subsequent runs can load directly and skip DTW computation

### 8. Performance Optimizations

| Technique | Effect |
|-----------|--------|
| **numba @njit** | All DP loops JIT-compiled to machine code, ~10ms per ~700-frame trajectory pair |
| **Sakoe-Chiba window** | Complexity reduced from O(N*M) to O(N*window) |
| **Multi-process parallelism** | `n_jobs` workers compute different trajectory pairs simultaneously |
| **Cache reuse** | Distance matrix saved as `.npz`; no recomputation when tuning parameters |

## File Structure

```
ClusteringAndSampling/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── config.py                              # Dataset configs (directory structure, field mapping, DTW parameters)
├── run_analysis.py                        # Single-dataset clustering analysis entry point
├── batch_run.py                           # Batch clustering analysis for all datasets
├── batch_run_by_task.py                   # Per-task clustering (for BC_Z/RT-1 etc.)
├── collect_cluster_representation.py      # Collect representative episodes per cluster
├── sample_filtered_episodes.py            # Random sampling from filtered datasets
└── utils/
    ├── __init__.py
    ├── trajectory_loader.py               # Trajectory data loading module
    ├── dtw_distance.py                    # DTW distance computation (numba JIT accelerated)
    ├── clustering_analysis.py             # Clustering algorithms + visualization
    └── sample_all.py                      # Sampling and record generation utilities
```

### `trajectory_loader.py` -- Trajectory Loading

- **`Trajectory` dataclass**: Encapsulates a single trajectory; core attribute `combined` is a `(T, 8)` matrix
  - Columns 0-2: xyz position
  - Columns 3-6: quaternion (qx, qy, qz, qw)
  - Column 7: gripper openness
- **`load_trajectories()`**: Scans `data/chunk-*/episode_*.parquet`, automatically reads `meta/modality.json` to map column names, splits trajectories by `episode_index`
- Supports `--side right/left` to switch between arms

### `dtw_distance.py` -- DTW Core

Each function corresponds to the sections in "DTW Algorithm in Detail" above:

| Function | Section | Description |
|----------|---------|-------------|
| `_quat_geodesic()` | S2 Per-frame cost | Quaternion geodesic angle `2*arccos(\|q1*q2\|)` |
| `frame_cost()` | S2 Per-frame cost | Weighted sum of position L2 + quaternion angle + gripper diff |
| `_dtw_cost_matrix()` | S3 Dynamic programming | Standard O(N*M) DP table fill |
| `_dtw_with_window()` | S4 Window constraint | Sakoe-Chiba band constraint, O(N*window) |
| `_backtrace_length()` | S5 Path backtracing | Backtrace path length from `dp[N][M]` for normalization |
| `dtw_distance()` | S6 Complete flow | Unified interface: with/without window + optional normalization |
| `compute_distance_matrix()` | S7 Batch matrix | N*N symmetric distance matrix with multi-process support |

All DP functions use `numba @njit(cache=True)` compilation; after the first JIT compilation the machine code is cached, subsequent calls take ~10ms per ~700-frame trajectory pair.

### `clustering_analysis.py` -- Clustering and Visualization

Clustering methods:

| Method | Function | Features |
|--------|----------|----------|
| Hierarchical | `hierarchical_clustering()` | Supports average/complete/single/ward, generates dendrogram |
| K-Medoids | `kmedoids_clustering()` | Directly based on distance matrix, outputs representative trajectory (medoid) per cluster |

Visualization outputs:

| Chart | Function | Description |
|-------|----------|-------------|
| Distance heatmap | `plot_distance_heatmap()` | N*N matrix sorted by cluster; same-cluster trajectories form color blocks |
| Dendrogram | `plot_dendrogram()` | Hierarchical clustering merge process |
| MDS scatter plot | `plot_mds_embedding()` | Distance matrix reduced to 2D, colored by cluster |

### `run_analysis.py` -- Main Entry Point

4-step pipeline:

```
[1/4] Load trajectories -> Read parquet, extract (T, 8) sequences
[2/4] DTW distances     -> Compute N*N distance matrix, auto-cached as .npz
[3/4] Clustering        -> Hierarchical clustering + K-Medoids, print cluster members
[4/4] Visualization     -> Output heatmap, dendrogram, MDS scatter plot
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python run_analysis.py \
    --dataset_root /path/to/Galaxea-Open-World-Dataset/Adjust_xxx \
    --side right \
    --n_clusters 5 \
    --output_dir ./results
```

### Speedup (Window Constraint + Multi-process)

```bash
python run_analysis.py \
    --dataset_root /path/to/dataset \
    --window 100 \
    --n_jobs 8 \
    --n_clusters 5 \
    --output_dir ./results
```

### Load from Cache (Skip DTW, Cluster Directly)

```bash
python run_analysis.py \
    --dataset_root /path/to/dataset \
    --load_cache ./results/distance_matrix.npz \
    --n_clusters 3
```

### Quick Debug (Only First 10 Trajectories)

```bash
python run_analysis.py \
    --dataset_root /path/to/dataset \
    --max_episodes 10 \
    --n_clusters 3 \
    --output_dir ./results_test
```

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_root` | Required | Dataset root directory (containing `data/` and `meta/`) |
| `--side` | `right` | Which arm to analyze: `right` or `left` |
| `--max_episodes` | All | Maximum number of trajectories to load |
| `--w_pos` | `1.0` | Position distance weight |
| `--w_rot` | `1.0` | Orientation distance weight |
| `--w_grip` | `0.5` | Gripper distance weight |
| `--normalize` | `True` | Normalize DTW distance by path length |
| `--window` | Unconstrained | Sakoe-Chiba window size (recommended 50~200) |
| `--n_jobs` | `1` | Number of parallel processes |
| `--n_clusters` | `5` | Number of clusters |
| `--cluster_method` | `both` | `hierarchical` / `kmedoids` / `both` |
| `--linkage_method` | `average` | Hierarchical clustering linkage: `average`/`complete`/`single`/`ward` |
| `--output_dir` | `./results` | Output directory |
| `--load_cache` | None | Load existing distance matrix `.npz` file |

## Output Files

| File | Description |
|------|-------------|
| `distance_matrix.npz` | N*N DTW distance matrix + episode_ids (reusable) |
| `cluster_labels.npz` | Cluster labels + medoid indices |
| `distance_heatmap.png` | Distance matrix heatmap |
| `dendrogram.png` | Hierarchical clustering dendrogram |
| `mds_embedding.png` | MDS 2D scatter plot |

## Data Format Requirements

The dataset directory structure must follow LeRobot v2.1 format:

```
dataset_root/
├── meta/
│   └── modality.json      # Field mapping (optional; defaults used if absent)
└── data/
    ├── chunk-000/
    │   ├── episode_000000.parquet
    │   ├── episode_000001.parquet
    │   └── ...
    └── chunk-001/
        └── ...
```

The parquet files must contain the following columns (using right arm as example):

| Column | Dimensions | Description |
|--------|-----------|-------------|
| `observation.state.right_ee_pose` | 7 | xyz(3) + quaternion_xyzw(4) |
| `observation.state.right_gripper` | 1 | Gripper openness |
| `episode_index` | 1 | Episode number (used for splitting trajectories) |
