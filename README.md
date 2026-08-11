# ChArUco 标定工具使用说明

本目录包含基于 OpenCV ChArUco 的相机标定工具，主要由 `generate_charuco.py`（生成标定板）和 `charucoCalibrate/aruco.py`（检测与位姿估计）两个模块组成。

---

## 1. generate_charuco.py — 生成 ChArUco 标定板

用于生成 ChArUco 标定板图像，可打印后用于相机标定。

### 核心函数

```python
def generate_charuco_board(
    squares_x=4,        # 水平方向方格数
    squares_y=6,        # 垂直方向方格数
    square_size=80,     # 每个方格的像素大小
    marker_ratio=0.75,  # ArUco 标记边长占方格边长的比例 (0.5~0.8)
    dict_id=cv.aruco.DICT_6X6_250  # ArUco 字典 ID
)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `squares_x` | `4` | 水平方向棋盘格数，建议 5~9 以获得更多角点 |
| `squares_y` | `6` | 垂直方向棋盘格数，建议 6~11 |
| `square_size` | `80` | 每个方格像素大小（最终图像尺寸 = squares × square_size） |
| `marker_ratio` | `0.75` | 标记尺寸 / 方格尺寸，推荐 0.5~0.8 |
| `dict_id` | `DICT_6X6_250` | 字典类型，影响支持的标记数量 |

### 使用示例

```bash
python generate_charuco.py
```

运行后会在当前目录生成 `charuco_board.png`，打印该图像即可作为标定板使用。

### 更换字典

如需更多标记数量，可替换 `dict_id`：

```python
# 常用字典选项:
cv.aruco.DICT_4X4_50    # 4×4 标记, 50 个
cv.aruco.DICT_5X5_100   # 5×5 标记, 100 个
cv.aruco.DICT_6X6_250   # 6×6 标记, 250 个 (默认)
cv.aruco.DICT_7X7_1000  # 7×7 标记, 1000 个
```

---

## 2. charucoCalibrate/aruco.py — ArUco / ChArUco 检测框架

提供面向对象的 ArUco/ChArUco 操作封装，包含四个类，继承关系为：

```
aruco  →  arucoBoard  →  charucoBoard  →  charucoDiamond
```

### 2.1 aruco — 单个 ArUco 标记

```python
from charucoCalibrate.aruco import aruco

detector = aruco(dictionary_id=cv2.aruco.DICT_6X6_50)
```

| 方法 | 说明 |
|------|------|
| `generate(id, size)` | 生成指定 ID 的 ArUco 标记图像 |
| `detect(image)` | 检测图像中的 ArUco 标记，返回 `corners, ids, rejected` |
| `pose_estimate(corners, cameraMatrix, distCoeffs)` | 根据角点计算标记位姿 (PnP)，返回 `{cameraMatrix, distCoeffs, rvec, tvec}` |
| `draw(image, corners, ids, pose, axis_size)` | 在图像上绘制检测到的标记和坐标轴 |

### 2.2 arucoBoard — ArUco 网格板

```python
board = arucoBoard(
    dictionary_id=cv2.aruco.DICT_6X6_50,
    board_size=(5, 7),           # 网格行列数
    marker_square_rate=0.1       # 标记占格子比例
)
```

继承了 `aruco` 的所有方法，并额外支持：
- `generate(size)` — 生成网格板图像
- `pose_estimate(corners, ids, cameraMatrix, distCoeffs)` — 使用板匹配点计算整体位姿

### 2.3 charucoBoard — ChArUco 板（推荐用于标定）

```python
from charucoCalibrate.aruco import charucoBoard

board = charucoBoard(
    dictionary_id=cv2.aruco.DICT_6X6_50,
    board_size=(5, 7),           # 棋盘格数 (cols, rows)
    marker_square_rate=0.6,      # 标记占格比例
    ids=None                     # 可指定标记 ID 列表
)
```

| 方法 | 说明 |
|------|------|
| `detect(image, refine)` | 检测 ChArUco 角点，返回 `charucoCorners, charucoIds, markerCorners, markerIds` |
| `pose_estimate(charucoCorners, charucoIds, cameraMatrix, distCoeffs)` | 基于角点匹配计算板位姿 |
| `draw(img, charucoCorners, charucoIds, pose, charuco_color, axis_size)` | 可视化检测结果和坐标轴 |

**ChArUco 相比普通棋盘格的优势：** 即使标定板部分被遮挡，仍可通过可见的 ArUco 标记恢复角点位置。

### 2.4 charucoDiamond — ChArUco 菱形检测

```python
diamond = charucoDiamond(
    ids=None,
    dictionary_id=cv2.aruco.DICT_6X6_50,
    marker_square_rate=0.6
)
```

用于检测四个 ArUco 标记围成的菱形（diamond）图案。

---

## 3. 完整标定流程

### 步骤 1：生成标定板

```bash
python generate_charuco.py
```

### 步骤 2：拍摄标定图像

用相机从不同角度拍摄 ChArUco 标定板（建议 15~30 张），放入 `charucoCalibrate/data/` 目录。

### 步骤 3：相机内参标定

```bash
cd charucoCalibrate
python calibration.py --path ./data --suffix .jpg --output ./output
```

标定完成后，`output/` 目录会生成：
- `camera_matrix.csv` — 相机内参矩阵 (3×3)
- `distortion_coefficients.csv` — 畸变系数

### 步骤 4：使用标定结果

```python
from charucoCalibrate.aruco import charucoBoard
import numpy as np

# 加载标定结果
camera_matrix = np.loadtxt('camera_matrix.csv', delimiter=',')
dist_coeffs = np.loadtxt('distortion_coefficients.csv', delimiter=',')

# 创建检测器
board = charucoBoard(board_size=(5, 7))

# 检测并估计位姿
charucoCorners, charucoIds, _, _ = board.detect(image)
if charucoIds is not None and len(charucoIds) > 4:
    pose = board.pose_estimate(charucoCorners, charucoIds, camera_matrix, dist_coeffs)
    print(f"旋转向量: {pose['rvec'].ravel()}")
    print(f"平移向量: {pose['tvec'].ravel()}")
```

---

## 依赖

```bash
pip install opencv-contrib-python numpy
```

> **注意：** 必须安装 `opencv-contrib-python` 而不是 `opencv-python`，因为 ArUco/ChArUco 模块在 contrib 包中。
