import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_moge_root := str(Path(__file__).absolute().parents[1] / "MoGe")) not in sys.path:
    sys.path.insert(0, _moge_root)

import cv2
import torch
import json
import numpy as np

from moge.model.v2 import MoGeModel

from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal # 确保导入 colorize_normal
import utils3d
from tqdm import tqdm
from dn_utils import intrinstics_from_camera_txt

def fov_from_intrinstics(a, width_pixels):
    """
    Args:
        a: 3*3
        width_pixels: 图像的宽度（像素）
    Output:
        fov_x (degrees)
    """
    fx = a[0, 0]
    fov_x_radians = 2 * np.arctan(width_pixels / (2 * fx))
    fov_x_degrees = fov_x_radians * (180 / np.pi)
    return fov_x_degrees

def extract(project_folder, device, resolution_level = 9, num_tokens = None, use_fp16 = False, output_normal = False):
    image_folder = Path(project_folder) / "images"
    save_path_root = image_folder.parent
    save_path_root.mkdir(exist_ok=True, parents=True)

    model_name = "Ruicheng/moge-2-vitl-normal"
    print(f"Loading model: {model_name}")
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    if use_fp16:
        model.half()

    depth_folder = Path(project_folder) / 'depth'
    normal_folder = Path(project_folder) / 'normal'

    image_paths = list(sorted(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))))
    print(f"Found images: {image_paths}")

    depth_folder.mkdir(exist_ok=True)
    if output_normal:
        normal_folder.mkdir(exist_ok=True)

    depth_numpy_array = None
    normal_numpy_array = None

    intrinstics = intrinstics_from_camera_txt(f"{project_folder}/sparse/0/cameras.txt")

    for idx, image_file in tqdm(enumerate(image_paths), total=len(image_paths)):
        image = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        current_intrinsics = intrinstics[idx]
        fov_x = fov_from_intrinstics(current_intrinsics, image_tensor.shape[2]) # image_tensor.shape[2] 是宽度
        output = model.infer(image_tensor, fov_x=fov_x, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
        
        if 'depth' not in output or output['depth'] is None:
            raise RuntimeError(f"Model '{model_name}' did not output 'depth' for image {image_file.name}.")
        current_depth_data = output['depth'].cpu().numpy()

        if(depth_numpy_array is None):
            depth_numpy_array = np.zeros((len(image_paths),) + current_depth_data.shape, dtype = np.float32)
        depth_numpy_array[idx] = current_depth_data

        if int(image_file.stem) <= 10:
            cv2.imwrite(str(depth_folder / f'{image_file.stem}.png'), cv2.cvtColor(colorize_depth(current_depth_data), cv2.COLOR_RGB2BGR))

        if output_normal:
            if 'normal' not in output or output['normal'] is None:
                raise RuntimeError(f"Model '{model_name}' was expected to output 'normal' but did not for image {image_file.name}.")
            current_normal_data = output['normal'].cpu().numpy()

            if(normal_numpy_array is None):
                normal_numpy_array = np.zeros((len(image_paths),) + current_normal_data.shape, dtype = np.float32)
            normal_numpy_array[idx] = current_normal_data

            if int(image_file.stem) <= 10:
                cv2.imwrite(str(normal_folder / f'{image_file.stem}.png'), cv2.cvtColor(colorize_normal(current_normal_data), cv2.COLOR_RGB2BGR))
    
    print(f"Generated depth numpy shape: {depth_numpy_array.shape}")
    np.save(str(project_folder / f'depth.npy'), depth_numpy_array)
    if output_normal:
        print(f"Generated normal numpy shape: {normal_numpy_array.shape}")
        np.save(str(project_folder / f'normal.npy'), normal_numpy_array)

if __name__ == "__main__":
    extract("/root/Projects/Supervised_Sugar/dataset/room_0/", torch.device("cuda:0"), output_normal=True)