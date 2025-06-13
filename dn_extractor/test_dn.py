import numpy as np
import torch

from pathlib import Path
import sys
if (_moge_root := str(Path(__file__).absolute().parents[1])) not in sys.path:
    sys.path.insert(0, _moge_root)
if (_moge_root := str(Path(__file__).absolute().parents[1] / "gaussian_splatting")) not in sys.path:
    sys.path.insert(0, _moge_root)
from sugar_scene.cameras import load_gs_cameras
from sugar_trainers.coarse_density_and_dn_consistency import depth_normal_consistency_loss

if __name__ == "__main__":
    project_folder = "/root/Projects/Supervised_Sugar/resources/truck"
    cam_list = load_gs_cameras(
        source_path=f"{project_folder}",
        gs_output_path=f"{project_folder}/sparse/0/",
    )
    
    depth_path = f"{project_folder}/depth/depth.npy"
    normal_path = f"{project_folder}/normal/normal.npy"

    depth = torch.tensor(np.load(depth_path)).cuda()  # (N, H, W)
    normal = torch.tensor(np.load(normal_path)).cuda()  # (N, H, W, 3)

    # 假设我们有一个 camera 对象
    camera = ...  # 请根据实际情况初始化 camera 对象

    # 计算平均 dn_consistency_loss
    losses = []
    for i in range(depth.shape[0]):
        depth_i = depth[i:i+1]  # 取第 i 个深度图
        normal_i = normal[i].permute(2, 0, 1)  # 转换法线图形状为 (3, H, W)
        loss = depth_normal_consistency_loss(depth_i, normal_i, camera)
        losses.append(loss.item())

    average_loss = sum(losses) / len(losses)
    print(f"Average dn_consistency_loss: {average_loss}")
