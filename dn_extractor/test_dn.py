import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt

# 保持 measure_depth_normal_consistency 函数不变，因为它是一个通用的计算逻辑
def measure_depth_normal_consistency(depth_map, normal_map, camera_intrinsics):
    """
    衡量深度图和法线图之间的一致性。

    Args:
        depth_map (np.ndarray): 深度图，形状为 (H, W)，通常为浮点类型（例如 np.float32），
                                 单位通常为米。
        normal_map (np.ndarray): 法线图，形状为 (H, W, 3)，通常为浮点类型（例如 np.float32）。
                                  每个像素包含 (Nx, Ny, Nz) 法线向量，范围通常在 [-1, 1]。
                                  请确保法线向量是单位向量。
        camera_intrinsics (dict): 包含相机内参的字典。
                                  例如：{'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    Returns:
        tuple: (consistency_map, mean_angle_error)
               - consistency_map (np.ndarray): 形状为 (H, W) 的数组，表示每个像素的一致性。
                                               通常是角度误差（弧度或度）。
               - mean_angle_error (float): 整个图像的平均角度误差（弧度或度）。
    """

    H, W = depth_map.shape
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # 1. 从深度图推导3D点云
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    x_cam = (u - cx) * depth_map / fx
    y_cam = (v - cy) * depth_map / fy
    z_cam = depth_map

    points_3d = np.stack((x_cam, y_cam, z_cam), axis=-1) # 形状 (H, W, 3)

    # 2. 从3D点云推导法线
    derived_normal_map = np.zeros_like(normal_map, dtype=np.float32)

    # 遍历图像，计算每个像素的法线
    for i in range(H - 1):
        for j in range(W - 1):
            p_curr = points_3d[i, j]
            p_right = points_3d[i, j + 1]
            p_down = points_3d[i + 1, j]

            # 检查深度是否有效，避免 NaN 或 Inf
            if np.isnan(p_curr).any() or np.isnan(p_right).any() or np.isnan(p_down).any():
                continue

            vec1 = p_right - p_curr
            vec2 = p_down - p_curr

            normal = np.cross(vec1, vec2)

            norm = np.linalg.norm(normal)
            if norm > 1e-6: # 避免除以零
                derived_normal_map[i, j] = normal / norm
            # else: 保持为零向量，表示无法计算有效法线

    # 3. 比较法线：计算角度差
    dot_product = np.sum(derived_normal_map * normal_map, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_difference_rad = np.arccos(dot_product)

    valid_mask = np.linalg.norm(derived_normal_map, axis=-1) > 1e-6
    consistency_map = np.full(depth_map.shape, np.nan, dtype=np.float32)
    consistency_map[valid_mask] = angle_difference_rad[valid_mask]

    # 4. 量化一致性：计算平均角度误差
    mean_angle_error = np.nanmean(consistency_map)

    return consistency_map, mean_angle_error

def calculate_consistency_for_image_0_from_real_path(project_folder):
    """
    加载指定 project_folder 中的真实数据，并计算第0张图片深度和法线的一致性。

    Args:
        project_folder (str): 项目文件夹的路径，例如 '/root/Projects/Supervised_Sugar/dataset/undistorted'。

    Returns:
        float: 第0张图片的平均角度误差（弧度）。如果数据加载失败，返回 None。
    """
    depth_path = os.path.join(project_folder, 'depth', 'depth.npy')
    normal_path = os.path.join(project_folder, 'normal', 'normal.npy')
    cameras_path = os.path.join(project_folder, 'sparse', '0', 'cameras.json')

    print(f"Attempting to load depth from: {depth_path}")
    print(f"Attempting to load normal from: {normal_path}")
    print(f"Attempting to load cameras from: {cameras_path}")

    # 检查文件是否存在
    if not os.path.exists(depth_path):
        print(f"Error: Depth file not found at {depth_path}. Please check the path.")
        return None
    if not os.path.exists(normal_path):
        print(f"Error: Normal file not found at {normal_path}. Please check the path.")
        return None
    if not os.path.exists(cameras_path):
        print(f"Error: Cameras file not found at {cameras_path}. Please check the path.")
        return None

    try:
        # 加载数据
        all_depth_maps = np.load(depth_path) # (N, H, W)
        all_normal_maps = np.load(normal_path) # (N, H, W, 3)
        print(all_depth_maps.shape, all_normal_maps.shape)
        # 计算 NaN 和 Inf 的数量
        nan_count_depth = np.isnan(all_depth_maps).sum()
        inf_count_depth = np.isinf(all_depth_maps).sum()
        nan_count_normal = np.isnan(all_normal_maps).sum()
        inf_count_normal = np.isinf(all_normal_maps).sum()

        print(f"Depth maps - NaN count: {nan_count_depth}, Inf count: {inf_count_depth}")
        print(f"Normal maps - NaN count: {nan_count_normal}, Inf count: {inf_count_normal}")

        with open(cameras_path, 'r') as f:
            cameras_data = json.load(f)

        # 确保 cameras.json 有相机数据，并获取第0个相机
        if not cameras_data or 'cameras' not in cameras_data or not cameras_data['cameras']:
            print("Error: 'cameras' list is empty or missing in cameras.json.")
            return None

        # 假设 cameras 列表的第一个元素就是对应第0张图片的相机信息
        camera_info = cameras_data['cameras'][0]

        # 提取相机内参
        if camera_info['model'] != "PINHOLE" or len(camera_info['params']) != 4:
            print(f"Error: Unsupported camera model '{camera_info['model']}' or invalid params length for camera_id {camera_info['camera_id']}")
            return None

        fx, fy, cx, cy = camera_info['params']
        camera_intrinsics = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }

        # 选择第0张图片的数据
        if all_depth_maps.shape[0] == 0 or all_normal_maps.shape[0] == 0:
            print("Error: No images found in depth.npy or normal.npy.")
            return None
        
        if all_depth_maps.shape[0] <= 0 or all_normal_maps.shape[0] <= 0:
            print("Error: Depth or normal data is empty.")
            return None


        depth_map_0 = all_depth_maps[0]
        normal_map_0 = all_normal_maps[0]

        # 检查尺寸是否匹配
        expected_H, expected_W = camera_info['height'], camera_info['width']
        if depth_map_0.shape != (expected_H, expected_W):
            print(f"Warning: Depth map (Image 0) shape {depth_map_0.shape} does not match camera resolution ({expected_H}, {expected_W}) from cameras.json.")
            print("Proceeding with actual image dimensions for consistency calculation, but this might indicate a data mismatch.")
            # 如果实际图像尺寸和相机内参尺寸不一致，通常意味着数据有问题，或者需要对图像进行resize。
            # 这里我们继续使用加载的图像尺寸，并调整cx,cy，以避免索引越界。
            # 更严谨的做法是强制要求尺寸匹配或者进行resize。
            if camera_intrinsics['cx'] >= depth_map_0.shape[1] or camera_intrinsics['cy'] >= depth_map_0.shape[0]:
                print("Warning: Camera principal point (cx, cy) is outside actual image dimensions. Adjusting to actual image center.")
                camera_intrinsics['cx'] = depth_map_0.shape[1] / 2.0
                camera_intrinsics['cy'] = depth_map_0.shape[0] / 2.0

        if normal_map_0.shape[:2] != depth_map_0.shape:
            print(f"Error: Normal map (Image 0) shape {normal_map_0.shape} does not match depth map shape {depth_map_0.shape}.")
            return None

        # 计算一致性
        consistency_map, mean_error = measure_depth_normal_consistency(
            depth_map_0, normal_map_0, camera_intrinsics
        )

        print(f"\n--- Consistency for Image 0 ---")
        print(f"Mean angle error (radians): {mean_error:.4f}")
        print(f"Mean angle error (degrees): {np.degrees(mean_error):.2f}")

        # 可视化 (可选)
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 4, 1)
        plt.imshow(depth_map_0, cmap='viridis')
        plt.title('Depth Map (Image 0)')
        plt.colorbar()

        plt.subplot(1, 4, 2)
        plt.imshow((normal_map_0 + 1) / 2) # 将法线从 [-1, 1] 映射到 [0, 1]
        plt.title('Predicted Normal Map (Image 0)')

        plt.subplot(1, 4, 3)
        consistency_map_display = np.copy(consistency_map)
        consistency_map_display[np.isnan(consistency_map_display)] = 0 # 将 NaN 值设置为0或一个特定颜色
        plt.imshow(np.degrees(consistency_map_display), cmap='plasma_r', vmin=0, vmax=90) # 角度误差，0-90度
        plt.title('Consistency Map (Angle Error in Degrees)')
        plt.colorbar(label='Angle Error (Degrees)')

        plt.subplot(1, 4, 4)
        # 为了可视化推导出的法线，这里重新计算一次
        derived_normal_map_display = np.zeros_like(normal_map_0)
        H_temp, W_temp = depth_map_0.shape
        u_temp, v_temp = np.meshgrid(np.arange(W_temp), np.arange(H_temp))
        x_cam_temp = (u_temp - camera_intrinsics['cx']) * depth_map_0 / camera_intrinsics['fx']
        y_cam_temp = (v_temp - camera_intrinsics['cy']) * depth_map_0 / camera_intrinsics['fy']
        z_cam_temp = depth_map_0
        points_3d_temp = np.stack((x_cam_temp, y_cam_temp, z_cam_temp), axis=-1)

        for i in range(H_temp - 1):
            for j in range(W_temp - 1):
                p_curr = points_3d_temp[i, j]
                p_right = points_3d_temp[i, j + 1]
                p_down = points_3d_temp[i + 1, j]
                if np.isnan(p_curr).any() or np.isnan(p_right).any() or np.isnan(p_down).any():
                    continue
                vec1 = p_right - p_curr
                vec2 = p_down - p_curr
                normal = np.cross(vec1, vec2)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    derived_normal_map_display[i, j] = normal / norm
        plt.imshow((derived_normal_map_display + 1) / 2)
        plt.title('Derived Normal Map (Image 0)')


        plt.tight_layout()
        plt.show()

        return mean_error

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# 主执行块
if __name__ == "__main__":
    # 使用您提供的真实项目路径
    real_project_folder = "/root/Projects/Supervised_Sugar/dataset/undistorted"

    print(f"Attempting to calculate consistency for image 0 from real path: {real_project_folder}")
    consistency_loss = calculate_consistency_for_image_0_from_real_path(real_project_folder)

    if consistency_loss is not None:
        print(f"Final consistency loss for image 0: {consistency_loss:.4f} radians ({np.degrees(consistency_loss):.2f} degrees)")
    else:
        print("Failed to calculate consistency loss. Please check error messages above.")