# 更新：增加 MoGe 作为 depth/normal extractor，使用 colmap 的 camera instrinstics 批量生成

## MoGe

[MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision](https://github.com/microsoft/MoGe)

请导入 submodule:

```sh
git submodule update
```

并按照其说明在其文件夹下

```sh
pip install -r requirements.txt
```

请注意，它输出的 depth 可能有 inf，这是因为它检测到那个 pixel 属于 unbounded 区域（比如天空）

normal 目前来看并没有。

## dn_extractor

- test_dn.py：计算 depth 和 normal 之间的 consistency

方法：depth -> point map -> normal

- dn_extractor/moge_tot.py：对一个 colmap project，批量操作

```sh
python moge_tot.py
```

要在程序的 main 中写好（`extract(path, torch.device("cuda:0"))`），这个 path 要有 images 子文件夹和 sparse 子文件夹

- dn_extractor/moge.py:

```python
def extract(image_path, output_path, device, fov_x_ = None, resolution_level = 9, num_tokens = None, use_fp16 = False):
```

后面的参数不用管，image_path 是一张图片的路径，output_path 是一个文件夹，可以是相对路径（如果是，那么相对于 image_path 的父文件夹）或绝对路径。

第一次运行时会通过 huggingface 下载它用的模型

- [DEPRECATED] dn_extractor/dsine_tot.py：对一个 colmap project，批量操作

```sh
python dsine_tot.py dsine.txt
```

## [DEPRECATED] DSINE

[Rethinking Inductive Biases for Surface Normal Estimation](https://github.com/baegwangbin/DSINE.git)

只需要

```sh
pip install geffnet
```
