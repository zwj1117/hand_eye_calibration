import mujoco
import mujoco.viewer
import cv2
import numpy as np
import time
import math
from pynput import keyboard
import threading

# ==========================================
# 1. 配置参数 (适配 Charuco Board)
# ==========================================
# XML文件路径
XML_PATH = "arx_l5.xml"

# --- Charuco 板参数 ---
# 原代码为 3列 x 5行 内角点，对应 4x6 个方块
SQUARES_X = 4
SQUARES_Y = 6
SQUARE_SIZE = 0.2  # 每个方块的物理尺寸 (米)
MARKER_SIZE = 0.15 # Marker 的物理尺寸 (米)，通常为方块尺寸的 75% 左右

# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# charuco.py 中改为：
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # ← 与生成脚本一致


charuco_board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE,
    MARKER_SIZE,
    ARUCO_DICT
)

# 新版 API: CharucoDetector 需要两个不同类型的参数对象
CHARUCO_PARAMS = cv2.aruco.CharucoParameters()    # Charuco 专用参数
ARUCO_PARAMS = cv2.aruco.DetectorParameters()      # ArUco 码检测参数
CHARUCO_DETECTOR = cv2.aruco.CharucoDetector(charuco_board, CHARUCO_PARAMS, ARUCO_PARAMS)


# --- 计算坐标偏移 (将原点移至中心) ---
# CharucoBoard 的 chessboardCorners 生成的坐标是从 (0,0,0) 开始的
# 我们需要计算中心偏移量，使其与原始物理配置一致
# 宽度方向有 (SQUARES_X - 1) 个间隔
offset_x = (SQUARES_X - 1) * SQUARE_SIZE / 2
offset_y = (SQUARES_Y - 1) * SQUARE_SIZE / 2

# 相机内参 (保持不变)
CAMERA_MATRIX = np.array([
    [634.18, 0, 640],
    [0, 649.34, 360],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# ==========================================
# 2. 按键监听辅助类 (保持不变)
# ==========================================
class KeyListener:
    def __init__(self):
        self.keys = {
            keyboard.Key.up: False, keyboard.Key.down: False,
            keyboard.Key.left: False, keyboard.Key.right: False,
            keyboard.Key.page_up: False, keyboard.Key.page_down: False,
            keyboard.Key.space: False,  # 采集图像
            keyboard.Key.enter: False   # 执行标定
        }
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if key in self.keys:
            self.keys[key] = True

    def on_release(self, key):
        if key in self.keys:
            self.keys[key] = False

    def stop(self):
        self.listener.stop()

# ==========================================
# 3. 主程序类
# ==========================================
class HandEyeCalibrationEnv:
    def __init__(self, xml_path):
        self.path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.key_listener = KeyListener()

        # 渲染器配置
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self.cam_name = "wrist_cam"

        # 机器人初始位姿
        self.data.qpos[:] = [0, 0.251, 0.314, 0, 0, 0]
        self.target_qpos = np.array([0, 0.251, 0.314, 0, 0, 0], dtype=np.float64)

        # 标定数据存储
        self.R_gripper2base_list = []
        self.t_gripper2base_list = []
        self.R_target2cam_list = []
        self.t_target2cam_list = []
        self.calib_done = False

    def euler2mat(self, euler):
        """欧拉角转旋转矩阵"""
        roll, pitch, yaw = euler
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    def get_true_cam_pose_from_xml(self):
        """从 XML 文件中读取真实的相机位姿"""
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.path)
        root = tree.getroot()
        T_true = None
        for body in root.iter('body'):
            if body.get('name') == 'link6':
                for cam in body.iter('camera'):
                    if cam.get('name') == 'wrist_cam':
                        pos = [float(x) for x in cam.get('pos').split()]
                        euler = [float(x) for x in cam.get('euler').split()]
                        R = self.euler2mat(euler)
                        T_true = np.eye(4)
                        T_true[:3, :3] = R
                        T_true[:3, 3] = np.array(pos)
                        return T_true
        return None

    def get_robot_pose(self):
        link6_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        pos = self.data.xpos[link6_id].copy()
        mat = self.data.xmat[link6_id].reshape(3, 3).copy()
        return mat, pos

    def control_robot(self):
        """根据按键控制机械臂关节"""
        step = 0.01
        if self.key_listener.keys[keyboard.Key.left]: self.target_qpos[0] -= step
        if self.key_listener.keys[keyboard.Key.right]: self.target_qpos[0] += step
        if self.key_listener.keys[keyboard.Key.up]: self.target_qpos[1] -= step
        if self.key_listener.keys[keyboard.Key.down]: self.target_qpos[1] += step
        if self.key_listener.keys[keyboard.Key.page_up]: self.target_qpos[2] -= step
        if self.key_listener.keys[keyboard.Key.page_down]: self.target_qpos[2] += step

        for i in range(6):
            diff = self.target_qpos[i] - self.data.qpos[i]
            self.data.qpos[i] += diff * 0.01

    def collect_data(self, image):
        """检测 Charuco 角点并保存标定数据"""
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 使用新版 CharucoDetector 直接检测 Marker 和 Charuco 角点
        # 返回值: charuco_corners, charuco_ids, marker_corners, marker_ids
        charuco_corners, charuco_ids, marker_corners, marker_ids = CHARUCO_DETECTOR.detectBoard(gray)

        # 判断是否检测到足够的角点
        if charuco_ids is not None and len(charuco_ids) > 0:
            ret = len(charuco_ids)  # 检测到的角点数量

            # 至少需要检测到一定数量的角点才进行计算 (例如 4 个)
            if ret >= 6:
                # --- 获取对应的 3D 世界坐标 ---
                # 新版 API: 使用 getChessboardCorners() 方法替代 chessboardCorners 属性
                all_corners = charuco_board.getChessboardCorners()
                charuco_ids_flat = charuco_ids.flatten().astype(int)
                obj_points = all_corners[charuco_ids_flat].reshape(-1, 3)

                # --- 应用中心偏移 (保持原点在棋盘格中心) ---
                obj_points[:, 0] -= offset_x
                obj_points[:, 1] -= offset_y

                # --- 可视化 ---
                img_show = cv2.aruco.drawDetectedCornersCharuco(
                    image.copy(), charuco_corners, charuco_ids, (0, 255, 0)
                )
                cv2.imshow('Calibration View', img_show)
                cv2.waitKey(100)

                # --- PnP 求解 ---
                success, rvec, tvec = cv2.solvePnP(
                    obj_points.astype(np.float32),
                    charuco_corners.astype(np.float32),
                    CAMERA_MATRIX,
                    DIST_COEFFS
                )

                if success:
                    R_target2cam, _ = cv2.Rodrigues(rvec)
                    t_target2cam = tvec

                    # 获取机器人姿态
                    R_gripper2base, t_gripper2base = self.get_robot_pose()

                    # 保存数据
                    self.R_gripper2base_list.append(R_gripper2base)
                    self.t_gripper2base_list.append(t_gripper2base)
                    self.R_target2cam_list.append(R_target2cam)
                    self.t_target2cam_list.append(t_target2cam)

                    print(f"[采集成功 - Charuco] 当前已采集 {len(self.R_gripper2base_list)} 组数据 (检测到 {ret} 个角点)")
                else:
                    print("PnP 求解失败")
            else:
                print(f"检测到的角点不足 (当前: {ret})，请移动相机使更多区域可见。")
                cv2.imshow('Calibration View', gray)
                cv2.waitKey(10)
        else:
            print("未检测到 Charuco 角点")
            cv2.imshow('Calibration View', gray)
            cv2.waitKey(10)


    def run_calibration(self):
        if len(self.R_gripper2base_list) < 5:
            print("数据不足，至少需要5组。")
            return

        print("\n开始执行手眼标定...")
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.R_gripper2base_list, self.t_gripper2base_list,
            self.R_target2cam_list, self.t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # 构建结果矩阵
        T_result = np.eye(4)
        T_result[:3, :3] = R_cam2gripper
        T_result[:3, 3] = t_cam2gripper.flatten()

        print("\n================ 标定结果 (OpenCV 约定) ================")
        print("计算出的矩阵:")
        print(T_result)

        # 坐标系转换：OpenCV -> MuJoCo
        R_opencv2mujoco = np.diag([1, -1, -1])
        T_result_mujoco = T_result.copy()
        T_result_mujoco[:3, :3] = T_result[:3, :3] @ R_opencv2mujoco

        print("\n[转换后] MuJoCo 约定下的计算矩阵:")
        print(T_result_mujoco)

        # 验证
        T_true = self.get_true_cam_pose_from_xml()
        if T_true is not None:
            print("\n[验证] XML 中的真实矩阵:")
            print(T_true)

            T_diff = np.linalg.inv(T_true) @ T_result_mujoco
            R_diff = T_diff[:3, :3]
            trace = np.trace(R_diff)
            trace = max(-1.0, min(3.0, trace))
            theta_rad = math.acos((trace - 1) / 2)
            angle_error = math.degrees(theta_rad)
            trans_error = np.linalg.norm(T_diff[:3, 3])

            print("\n[验证] 误差分析:")
            print(f" 旋转误差: {angle_error:.4f} 度")
            print(f" 平移误差: {trans_error:.5f} 米")
            if angle_error < 1.0 and trans_error < 0.01:
                print(" -> 评价: 精度极高！")
            elif angle_error < 5.0 and trans_error < 0.05:
                print(" -> 评价: 精度良好。")
            else:
                print(" -> 警告: 误差较大。")
        else:
            print("\n[警告] 未能从 XML 读取真实矩阵。")

    def run(self):
        print("启动仿真...")
        print("控制说明:")
        print(" [方向键] 控制关节1/2 (旋转/俯仰)")
        print(" [PageUp/Down] 控制关节3 (俯仰)")
        print(" [空格] 采集当前视角数据")
        print(" [回车] 执行标定计算")
        
        cv2.namedWindow('Calibration View', cv2.WINDOW_NORMAL)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            while viewer.is_running():
                step_start = time.time()
                
                self.control_robot()
                mujoco.mj_step(self.model, self.data)
                
                self.renderer.update_scene(self.data, camera=self.cam_name)
                image = self.renderer.render()
                
                if self.key_listener.keys[keyboard.Key.space]:
                    self.collect_data(image)
                    self.key_listener.keys[keyboard.Key.space] = False
                
                if self.key_listener.keys[keyboard.Key.enter]:
                    self.run_calibration()
                    self.key_listener.keys[keyboard.Key.enter] = False
                
                viewer.sync()
                
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        self.key_listener.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    env = HandEyeCalibrationEnv(XML_PATH)
    env.run()
