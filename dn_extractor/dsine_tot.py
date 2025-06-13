import os
import sys
import glob
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../DSINE'))
import projects.dsine.config as config
import utils.utils as utils
from utils.projection import intrins_from_fov, intrins_from_txt
from tqdm import tqdm
from dn_utils import intrinstics_from_camera_txt

def extract(project_folder, device):
    args = config.get_args(test=True)
    print("Checkpoint from ", args.ckpt_path)
    assert os.path.exists(args.ckpt_path)

    if args.NNET_architecture == 'v00':
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == 'v01':
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == 'v02':
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == 'v02_kappa':
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception('invalid arch')
    
    model = DSINE(args).to(device)
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()

    img_paths = list(sorted(list(glob.glob(f'{project_folder}/images/*.jpg')) + list(glob.glob(f'{project_folder}/images/*.png'))))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm_numpy = []

    intrinstics = intrinstics_from_camera_txt(f"{project_folder}/sparse/0/cameras.txt")

    os.makedirs(f"{project_folder}/normal", exist_ok = True)

    with torch.no_grad():
        for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
            ext = os.path.splitext(img_path)[1]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = utils.get_padding(orig_H, orig_W)
            img = F.pad(img, lrtb, mode="constant", value=0.0)
            img = normalize(img)

            intrins = torch.tensor(intrinstics[idx], device=device).unsqueeze(0)
            
            intrins[:, 0, 2] += lrtb[0]
            intrins[:, 1, 2] += lrtb[2]

            pred_norm = model(img, intrins=intrins)[-1]
            pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            # save to norm_numpy
            pred_norm_float32 = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
            norm_numpy.append(pred_norm_float32[0])  # 记录每张图像的预测结果

            # 仅保存前10张图像的 PNG
            if idx < 10:
                target_path = f"{project_folder}/normal/{os.path.basename(img_path).split('.')[0]}.png"
                pred_norm_uint8 = (((pred_norm_float32 + 1) * 0.5) * 255).astype(np.uint8)
                im = Image.fromarray(pred_norm_uint8[0, ...])
                im.save(target_path)

    norm_numpy = np.array(norm_numpy)  # 转换为 NumPy 数组，形状为 (total_image_num, H, W, C)
    print(norm_numpy.shape)
    np.save(f'{project_folder}/normal/normal.npy', norm_numpy)


if __name__ == '__main__':
    extract('/root/Projects/Supervised_Sugar/resources/truck/', torch.device("cuda:0"))
