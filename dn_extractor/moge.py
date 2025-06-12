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

def extract(image_path, output_path, device, fov_x_ = None, resolution_level = 9, num_tokens = None, use_fp16 = False):
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    output = model.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
    points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)

    save_path = Path(image_path).parent / output_path
    save_path.mkdir(exist_ok=True, parents=True)

    cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_path / 'normal_vis.png'), cv2.cvtColor(colorize_normal(normals, normals_mask), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_path / 'depth.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    cv2.imwrite(str(save_path / 'mask.png'), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    with open(save_path / 'fov.json', 'w') as f:
        json.dump({
            'fov_x': round(float(np.rad2deg(fov_x)), 2),
            'fov_y': round(float(np.rad2deg(fov_y)), 2),
        }, f)

if __name__ == "__main__":
    extract("/root/Projects/Supervised_Sugar/resources/truck/images/000001.jpg", "./000001", "cuda:0")