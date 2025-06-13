# 更新：增加 MoGe 作为 depth/normal extractor.

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

## dn_extractor

我增加了 MoGe submodule 和 dn_extractor/moge.py，它有一个函数

```python
def extract(image_path, output_path, device, fov_x_ = None, resolution_level = 9, num_tokens = None, use_fp16 = False):
```

后面的参数不用管，image_path 是一张图片的路径，output_path 是一个文件夹，可以是相对路径（如果是，那么相对于 image_path 的父文件夹）或绝对路径。

第一次运行时会通过 huggingface 下载它用的模型