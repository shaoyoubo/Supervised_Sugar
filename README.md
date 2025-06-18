# Abstract

[3D Gaussian Splatting (3DGS)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) has recently become a powerful approach for multiview 3D reconstruction and novel-view synthesis, offering photorealistic rendering and efficient training through differentiable rendering of 3D Gaussians. While 3DGS excels at generating high-quality images, extracting accurate surface meshes remains a significant challenge due to the sparsity of the underlying density field and the lack of geometric priors. Recent methods such as [SuGaR](https://anttwo.github.io/sugar/) address this by introducing surface-aligned regularization, but their reliance on self-supervised normal estimation can lead to suboptimal performance in complex scenes. In this work, we propose a hybrid regularization framework that enhances 3DGS meshability by combining regularization from [SuGaR](https://anttwo.github.io/sugar/) with external normal and depth supervision. Our training strategy gradually transitions from external guidance to self-supervision, improving geometric consistency while preserving rendering fidelity. Experimental results demonstrate that our approach produces smoother and more accurate surface meshes, bridging the gap between photometric reconstruction and geometric understanding.

# BibTeX

```
@software{shao2025spsugar,
  author = {Shao, Youbo and Wu, Yuyang and Zhang, Kaixin},
  title = {Supervised SuGaR: Supervised Surface Aligned Gaussian Splatting},
  url = {https://github.com/shaoyoubo/Supervised_Sugar},
  year = {2025},
  publisher = {GitHub}
}
```

# Overview



# Installation

Follow the installation of [SuGaR](https://anttwo.github.io/sugar/).

For dataset generation, run `git submodule init` to download [MoGe](https://github.com/microsoft/MoGe.git).

# Usage

## Dataset

1. Start from a [COLMAP](https://github.com/colmap/colmap) dataset.

2. Set correct folder path in `dn_extractor/moge_tot.py`, then run `dn_extractor/moge_tot.py`.

The folder should have `images` subfolder (multi-view images) and `sparse/0/` subfolder (sparse reconstruction of COLMAP).

This script will generate `depth.npy` (estimated depth map) and `normal.npy` (estimated normal map) under the folder.

3. If you want to reproduce our result, download Truck dataset from [Tanks&Temples](https://www.tanksandtemples.org/download/), resize the images to $360 \times 640$, save to `dataset/resized`, then follow the above steps.

## Training

Run `script/supervised_full_line.sh`. It runs for five different weighting functions.

## Evaluation

Run `script/metric_3dgs.sh` for one specific configuration.

## Visualization

See `render_normal.ipynb` and `render_no_texture_mesh.ipynb`

