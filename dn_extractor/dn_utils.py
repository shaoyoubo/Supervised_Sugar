import numpy as np
import math, os, json, torch
import PIL.Image as Image

from pathlib import Path
import sys
if (_moge_root := str(Path(__file__).absolute().parents[1])) not in sys.path:
    sys.path.insert(0, _moge_root)
from sugar_scene.cameras import load_gs_cameras

def intrinstics_from_camera_txt(intrinstics_path):
    """
    Args:
        file_path (str): path of cameras.txt

    Returns:
        numpy.ndarray: N*3*3
    """
    intrinsics_list = []
    with open(intrinstics_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8: # ID, MODEL, W, H, fx, fy, cx, cy
                # print(f"Skipping malformed line: {line}")
                continue
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            if model == "PINHOLE":
                try:
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    intrinsics_list.append(K)
                    #intrinsics_list.append([fx, fy, cx, cy])
                except ValueError:
                    continue
            else:
                pass
    return np.array(intrinsics_list, dtype=np.float32)

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

