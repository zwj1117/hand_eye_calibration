# ChArUco 标定工具使用说明

本目录包含基于 OpenCV ChArUco 的相机标定工具，主要由 `generate_charuco.py`（生成标定板）和 `charucoCalibrate/aruco.py`（标定代码）两个模块组成。

---

## 依赖

```bash
pip install opencv-contrib-python numpy
```

> **注意：** 必须安装 `opencv-contrib-python` 而不是 `opencv-python`，因为 ArUco/ChArUco 模块在 contrib 包中。
