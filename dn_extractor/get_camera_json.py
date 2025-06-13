import json

# 定义 json 中单个相机的结构
def camera_to_dict(cam_id, model, width, height, params):
    return {
        "camera_id": cam_id,
        "model": model,
        "width": width,
        "height": height,
        "params": params
    }

def read_cameras_txt(path):
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = camera_to_dict(cam_id, model, width, height, params)
    return cameras

def write_cameras_json(cameras, out_path):
    # 输出为 list 格式，方便后续使用
    data = {"cameras": list(cameras.values())}
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    project_folder = "/root/Projects/Supervised_Sugar/resources/truck"
    cams = read_cameras_txt(f"{project_folder}/sparse/0/cameras.txt")
    write_cameras_json(cams, f"{project_folder}/sparse/0/cameras.json")
    print(f"已生成 {len(cams)} 个相机，保存至 cameras.json")