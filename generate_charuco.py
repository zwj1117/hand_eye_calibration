import cv2 as cv
import numpy as np

def generate_charuco_board(squares_x=4, squares_y=6, square_size=80, marker_ratio=0.75, dict_id=cv.aruco.DICT_6X6_250):
    """
    生成 ChArUco 标定板
    :param squares_x: 水平方向方格数 (棋盘格交点数 = squares_x - 1)
    :param squares_y: 垂直方向方格数 (棋盘格交点数 = squares_y - 1)
    :param square_size: 每个方格的像素大小
    :param marker_ratio: 标记边长占方格边长的比例 (通常在 0.5~0.8 之间)
    :param dict_id: ArUco 字典 ID (如 DICT_4X4_50, DICT_6X6_250 等)
    :return: ChArUco 板图像
    """
    # 1. 定义 ArUco 字典
    aruco_dict = cv.aruco.getPredefinedDictionary(dict_id)
    
    # 2. 创建 ChArUco 板对象
    # 参数: (X方向方格数, Y方向方格数), 方格尺寸, 标记尺寸, 字典
    board = cv.aruco.CharucoBoard((squares_x, squares_y), square_size, square_size * marker_ratio, aruco_dict)
    
    # 3. 计算生成图像的尺寸
    # 宽度和高度应能容纳所有方格，并留出一点边距
    img_width = squares_x * square_size
    img_height = squares_y * square_size
    
    # 4. 生成图像
    # marginSize: 边距大小（像素），防止边缘被切掉
    # borderBits: 标记边界的位数，通常为1
    charuco_image = board.generateImage((img_width, img_height), marginSize=10, borderBits=1)
    
    return charuco_image

# 生成 ChArUco 板
# 建议方格数稍微多一点效果更好，例如 8x11，这里保留原参数 4x6 作为示例
charuco_board = generate_charuco_board(squares_x=4, squares_y=6, square_size=80)

# 保存图像
cv.imwrite('charuco_board.png', charuco_board)
print("✅ ChArUco 标定板已保存为 charuco_board.png")
print(f"图像尺寸: {charuco_board.shape}")
