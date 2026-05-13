# VLA 轨迹相似度分析与聚类

基于 **DTW（Dynamic Time Warping）** 对 VLA 数据集中的 EEF 位姿轨迹做**时间弹性对齐**，忽略动作快慢差异，只关注轨迹的空间形状与位姿序列，从而分析不同轨迹之间的相似性并进行聚类。

## 核心思路

### 为什么不能直接逐帧比较？

同一个动作可能做得快或慢，甚至**局部快慢不一致**（A 轨迹的中间一段比 B 快，另一段又比 B 慢）。直接逐帧对比会因为时间轴不对齐而得到错误的距离。

### DTW 如何解决？

DTW 在两条轨迹之间寻找一条**非线性的对齐路径**：
- 某一段 A 多帧对 B 一帧（A 这段更慢）
- 另一段 A 一帧对 B 多帧（A 这段更快）
- 再一段一帧对一帧

对齐是**逐段、逐位置**的，不是整条轨迹只用一个缩放系数，因此能处理局部快慢不一致的情况。

## DTW 算法详解

### 1. 问题定义

给定两条轨迹：
- 轨迹 A：`a₁, a₂, ..., aₙ`（N 帧）
- 轨迹 B：`b₁, b₂, ..., bₘ`（M 帧）

每帧是一个 8 维向量 `[x, y, z, qx, qy, qz, qw, gripper]`。

DTW 的目标是找到一条**对齐路径（warp path）** `W = (w₁, w₂, ..., wₖ)`，其中每个 `wₜ = (iₜ, jₜ)` 表示"轨迹 A 的第 iₜ 帧 与 轨迹 B 的第 jₜ 帧对齐"，使得路径上所有帧对的代价之和最小。

### 2. 单帧代价函数

每对帧 `(aᵢ, bⱼ)` 的代价由三部分加权组合：

| 分量 | 公式 | 说明 |
|------|------|------|
| **位置** | `d_pos = sqrt((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)` | EEF 位置的欧氏距离（米） |
| **姿态** | `d_rot = 2·arccos(min(\|q₁·q₂\|, 1.0))` | 四元数测地线角距离（弧度） |
| **夹爪** | `d_grip = \|g₁ - g₂\|` | 夹爪开合度的差值 |

总代价：

```
cost(i, j) = w_pos × d_pos + w_rot × d_rot + w_grip × d_grip
```

**关于四元数测地线角距离**：

单位四元数 `q = (qx, qy, qz, qw)` 表示一个 3D 旋转。两个姿态之间的旋转角度等于相对旋转四元数的角度，而相对旋转四元数的标量部分恰好是两个四元数的 4D 点积。因此：

```
d_rot = 2 · arccos(|q₁ · q₂|)
```

- 取绝对值 `|q₁·q₂|` 是因为 `q` 与 `-q` 表示同一个旋转
- `clip` 到 1.0 是防止浮点误差导致 `arccos` 返回 NaN
- 值域为 `[0, π]`，即 0 到 180 度

### 3. 动态规划递推

DTW 使用动态规划求解最优对齐路径。定义 `dp[i][j]` 为轨迹 A 的前 i 帧与轨迹 B 的前 j 帧的最优累计代价。

**递推公式**：

```
dp[0][0] = 0
dp[i][j] = cost(i, j) + min(dp[i-1][j-1],   ← 对角线：一对一匹配
                             dp[i-1][j],       ← 垂直：A 的当前帧"重复"对齐 B 的多帧
                             dp[i][j-1])        ← 水平：B 的当前帧"重复"对齐 A 的多帧
```

**直观理解**——三种转移方向对应三种时间对齐方式：

```
           j (轨迹 B)
           →
    ┌──────────────────┐
    │  ╲  ← 对角(1:1)  │
  i │   ╲              │
(A) │    ╲             │
  ↓ │ ↓重复A帧         │
    │     →重复B帧      │
    └──────────────────┘
```

- **对角线 `dp[i-1][j-1]`**：A 的第 i 帧与 B 的第 j 帧一对一匹配，双方时间同步推进
- **垂直 `dp[i-1][j]`**：A 前进一帧，B 不动 —— 相当于 A 的这一段比 B 慢（A 的多帧对 B 的同一帧）
- **水平 `dp[i][j-1]`**：B 前进一帧，A 不动 —— 相当于 B 的这一段比 A 慢（B 的多帧对 A 的同一帧）

最终 `dp[N][M]` 就是两条轨迹的 DTW 原始距离。

**时间复杂度**：O(N × M)，对于 ~700 帧的轨迹，矩阵大小约 700×700 = 49 万个元素。

### 4. Sakoe-Chiba 窗口约束

无约束 DTW 允许任意弯曲的对齐路径，但：
- 计算量为完整的 O(N × M)
- 可能出现病态对齐（如 A 的第 1 帧匹配 B 的最后一帧）

**Sakoe-Chiba 窗口**限制对齐路径只能在对角线附近 `±w` 的带状区域内：

```
           j (轨迹 B)
    ┌──────────────────┐
    │╲▓▓▓              │   ▓ = 允许计算的区域
    │ ╲▓▓▓▓            │   （对角线 ± window）
    │  ╲▓▓▓▓           │
    │   ╲▓▓▓▓          │   空白 = 跳过，保持 ∞
    │    ╲▓▓▓▓         │
    │     ╲▓▓▓▓        │
    │      ╲▓▓▓        │
    └──────────────────┘
```

- 对齐路径上 `|i - j| ≤ w` 才计算，超出范围的 `dp[i][j]` 保持 `∞`
- 复杂度降为 **O(N × window)**，例如 window=100 时只需计算 700×200 = 14 万个元素
- 当 `|N - M| > window` 时，自动扩展窗口为 `max(window, |N - M|)` 以保证路径可达

### 5. 路径回溯与归一化

DTW 的原始距离 `dp[N][M]` 会随轨迹长度增长（路径越长，累加的代价越多）。为了使不同长度的轨迹对的距离可比较，需要做**归一化**。

**回溯**：从 `dp[N][M]` 出发，每步选择 `dp[i-1][j-1]`、`dp[i-1][j]`、`dp[i][j-1]` 中最小的前驱，直到回溯到 `dp[0][0]`，经过的步数即为路径长度 `L`。

```
归一化距离 = dp[N][M] / L
```

这样每对轨迹得到的是"平均每步代价"，消除了轨迹长度对距离的放大效应。

**带窗口模式**的归一化：由于不保存完整 dp 矩阵（只返回最终值），使用近似归一化 `dp[N][M] / (N + M)`，其中 `N + M` 是路径长度的上界。

### 6. 完整计算流程（以两条轨迹为例）

```
轨迹 A: (649, 8)   轨迹 B: (771, 8)
         │                    │
         ▼                    ▼
    ┌─────────────────────────────┐
    │  对每对 (i,j) 计算 frame_cost │  → 位置L2 + 四元数角 + 夹爪差
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  DP 填表: dp[i][j] =        │
    │    cost(i,j) + min(三个前驱) │  → O(649 × 771) ≈ 50万次
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  dp[649][771] = 原始DTW距离  │
    │  回溯路径长度 L ≈ 1000步     │
    │  归一化距离 = dp / L         │
    └─────────────────────────────┘
                  │
                  ▼
           距离 ≈ 0.329
```

### 7. 批量距离矩阵

对 N 条轨迹，需要计算 `N × (N-1) / 2` 个配对（对称矩阵只算上三角）。

- 100 条轨迹 → 4950 对
- 支持 `n_jobs > 1` 时通过 `multiprocessing.Pool` 并行加速
- 计算完成后自动缓存为 `.npz` 文件，后续可直接加载跳过 DTW 计算

### 8. 性能优化

| 手段 | 效果 |
|------|------|
| **numba @njit** | 所有 DP 循环 JIT 编译为机器码，~700 帧的轨迹对 ~10ms |
| **Sakoe-Chiba 窗口** | 复杂度从 O(N×M) 降到 O(N×window) |
| **多进程并行** | `n_jobs` 个 worker 同时计算不同轨迹对 |
| **缓存复用** | 距离矩阵存 `.npz`，调参时无需重算 |

## 文件结构

```
Clustering/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── trajectory_loader.py      # 轨迹数据加载模块
├── dtw_distance.py           # DTW 距离计算（numba JIT 加速）
├── clustering_analysis.py    # 聚类算法 + 可视化
└── run_analysis.py           # 命令行主入口
```

### `trajectory_loader.py` — 轨迹加载

- **`Trajectory` 数据类**：封装单条轨迹，核心属性 `combined` 为 `(T, 8)` 矩阵
  - 前 3 列：xyz 位置
  - 第 4-7 列：四元数 (qx, qy, qz, qw)
  - 第 8 列：夹爪开合度
- **`load_trajectories()`**：扫描 `data/chunk-*/episode_*.parquet`，自动读取 `meta/modality.json` 映射列名，按 `episode_index` 拆分轨迹
- 支持 `--side right/left` 切换左右臂

### `dtw_distance.py` — DTW 核心

各函数与上文"DTW 算法详解"各节对应：

| 函数 | 对应章节 | 说明 |
|------|----------|------|
| `_quat_geodesic()` | §2 单帧代价 | 四元数测地线角 `2·arccos(\|q₁·q₂\|)` |
| `frame_cost()` | §2 单帧代价 | 位置 L2 + 四元数角 + 夹爪差的加权和 |
| `_dtw_cost_matrix()` | §3 动态规划 | 标准 O(N×M) DP 填表 |
| `_dtw_with_window()` | §4 窗口约束 | Sakoe-Chiba 带状约束，O(N×window) |
| `_backtrace_length()` | §5 路径回溯 | 从 `dp[N][M]` 回溯路径长度用于归一化 |
| `dtw_distance()` | §6 完整流程 | 统一接口：选择有/无窗口 + 可选归一化 |
| `compute_distance_matrix()` | §7 批量矩阵 | N×N 对称距离矩阵，支持多进程并行 |

所有 DP 函数使用 `numba @njit(cache=True)` 编译，首次调用 JIT 编译后缓存机器码，后续调用 ~700 帧的轨迹对仅需 ~10ms

### `clustering_analysis.py` — 聚类与可视化

聚类方法：

| 方法 | 函数 | 特点 |
|------|------|------|
| 层次聚类 | `hierarchical_clustering()` | 支持 average/complete/single/ward，生成树状图 |
| K-Medoids | `kmedoids_clustering()` | 直接基于距离矩阵，输出每个簇的代表性轨迹（medoid） |

可视化输出：

| 图表 | 函数 | 说明 |
|------|------|------|
| 距离热力图 | `plot_distance_heatmap()` | N×N 矩阵，按聚类排序，同簇轨迹聚集成色块 |
| 树状图 | `plot_dendrogram()` | 层次聚类合并过程 |
| MDS 散点图 | `plot_mds_embedding()` | 距离矩阵降维到 2D，按聚类上色 |

### `run_analysis.py` — 主入口

4 步流水线：

```
[1/4] 加载轨迹    → 读 parquet，提取 (T, 8) 序列
[2/4] DTW 距离    → 计算 N×N 距离矩阵，自动缓存为 .npz
[3/4] 聚类        → 层次聚类 + K-Medoids，打印各簇成员
[4/4] 可视化      → 输出热力图、树状图、MDS 散点图
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方式

### 基本用法

```bash
python run_analysis.py \
    --dataset_root /path/to/Galaxea-Open-World-Dataset/Adjust_xxx \
    --side right \
    --n_clusters 5 \
    --output_dir ./results
```

### 加速（窗口约束 + 多进程）

```bash
python run_analysis.py \
    --dataset_root /path/to/dataset \
    --window 100 \
    --n_jobs 8 \
    --n_clusters 5 \
    --output_dir ./results
```

### 从缓存加载（跳过 DTW，直接聚类）

```bash
python run_analysis.py \
    --dataset_root /path/to/dataset \
    --load_cache ./results/distance_matrix.npz \
    --n_clusters 3
```

### 快速调试（只取前 10 条轨迹）

```bash
python run_analysis.py \
    --dataset_root /path/to/dataset \
    --max_episodes 10 \
    --n_clusters 3 \
    --output_dir ./results_test
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_root` | 必填 | 数据集根目录（包含 `data/` 和 `meta/`） |
| `--side` | `right` | 分析哪只手臂：`right` 或 `left` |
| `--max_episodes` | 全部 | 最多加载几条轨迹 |
| `--w_pos` | `1.0` | 位置距离权重 |
| `--w_rot` | `1.0` | 姿态距离权重 |
| `--w_grip` | `0.5` | 夹爪距离权重 |
| `--normalize` | `True` | DTW 距离按路径长度归一化 |
| `--window` | 无约束 | Sakoe-Chiba 窗口大小（推荐 50~200） |
| `--n_jobs` | `1` | 并行进程数 |
| `--n_clusters` | `5` | 聚类数 |
| `--cluster_method` | `both` | `hierarchical` / `kmedoids` / `both` |
| `--linkage_method` | `average` | 层次聚类 linkage：`average`/`complete`/`single`/`ward` |
| `--output_dir` | `./results` | 输出目录 |
| `--load_cache` | 无 | 加载已有距离矩阵 `.npz` 文件 |

## 输出文件

| 文件 | 说明 |
|------|------|
| `distance_matrix.npz` | N×N DTW 距离矩阵 + episode_ids（可复用） |
| `cluster_labels.npz` | 聚类标签 + medoid 索引 |
| `distance_heatmap.png` | 距离矩阵热力图 |
| `dendrogram.png` | 层次聚类树状图 |
| `mds_embedding.png` | MDS 2D 散点图 |

## 数据格式要求

数据集目录结构需为 LeRobot v2.1 格式：

```
dataset_root/
├── meta/
│   └── modality.json      # 字段映射（可选，无则使用默认列名）
└── data/
    ├── chunk-000/
    │   ├── episode_000000.parquet
    │   ├── episode_000001.parquet
    │   └── ...
    └── chunk-001/
        └── ...
```

parquet 中需包含以下列（以 right 为例）：

| 列名 | 维度 | 含义 |
|------|------|------|
| `observation.state.right_ee_pose` | 7 | xyz(3) + quaternion_xyzw(4) |
| `observation.state.right_gripper` | 1 | 夹爪开合度 |
| `episode_index` | 1 | episode 编号（用于拆分轨迹） |
