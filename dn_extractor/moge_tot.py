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

from moge.model.v1 import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
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

def extract(project_folder, device, resolution_level = 9, num_tokens = None, use_fp16 = False):
    image_folder = Path(project_folder) / "images"
    save_path = image_folder.parent
    save_path.mkdir(exist_ok=True, parents=True)

    depth_folder = Path(project_folder) / 'depth'
    image_paths = list(sorted(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))))
    print(image_paths)

    depth_folder.mkdir(exist_ok=True)
    depth_numpy = None
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    intrinstics = intrinstics_from_camera_txt(f"{project_folder}/sparse/0/cameras.txt")

    for idx, image_file in tqdm(enumerate(image_paths), total=len(image_paths)):
        image = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        fov_x = fov_from_intrinstics(intrinstics[idx], image_tensor.shape[1])
        output = model.infer(image_tensor, fov_x=fov_x, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
        
        depth = output['depth'].cpu().numpy()

        if(depth_numpy is None):
            depth_numpy = np.zeros((len(image_paths),) + depth.shape, dtype = np.float32)
        depth_numpy[idx] = depth

        if int(image_file.stem) <= 10:
            cv2.imwrite(str(depth_folder / f'{image_file.stem}.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
    
    print(depth_numpy.shape)
    np.save(str(depth_folder / f'depth.npy'), depth_numpy)

if __name__ == "__main__":
    extract("/root/Projects/Supervised_Sugar/resources/truck/", torch.device("cuda:0"))