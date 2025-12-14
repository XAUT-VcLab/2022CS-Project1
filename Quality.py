import os
import gc
import cv2
import trimesh
import pyrender
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 路径配置
reference_path = "/home/chensheng/1Project/Project1/FLAME/flame_zero_pose.obj"
gt_path = Path("/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/test_vertices")
output_path = Path("/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/_heatmap")
os.makedirs(output_path, exist_ok=True)

# 自定义颜色条：从蓝色到红色，去掉白色
blue_red_cmap = LinearSegmentedColormap.from_list("blue_red", ["red", "green", "blue"])

# 相机和渲染设置
cam = pyrender.PerspectiveCamera(yfov=np.pi / 20.0, aspectRatio=1.414)
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0]])
r = pyrender.OffscreenRenderer(960, 760)


def generate_heatmap():
    template_all = trimesh.load(reference_path).vertices.astype(np.float64)

    # 配置所有模型参数（路径+文件名处理方式）
    model_configs = [
        {  # STtb模型
            "name": "STtb",
            "path": Path("/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/STtb_vertices"),
        },
        {  # 其他模型（需要去掉test_前缀）
            "name": "CodeTalker",
            "path": Path("/home/chensheng/2Model/CodeTalker/RUN/mead/CodeTalker_s2_2/result/npy"),
        },
        {  # 其他模型（需要去掉test_前缀）
            "name": "FaceDiffuser",
            "path": Path("/home/chensheng/2Model/FaceDiffuser/result"),
        },
        {  # 其他模型（需要去掉test_前缀）
            "name": "FaceFormer",
            "path": Path("/home/chensheng/2Model/FaceFormer/mead/result"),
            "get_gt_name": lambda x: x.name.replace("test_", "", 1)
        },
        {  # 其他模型（需要去掉test_前缀）
            "name": "ProbTalk3D",
            "path": Path("/home/chensheng/2Model/ProbTalk3D/results/evaluation/vqvae_pred/2fiqeonq/0.2_multi/vert"),
        },
    ]

    # 以第一个模型（STtb）的文件列表为基准
    base_model = model_configs[0]
    all_base_files = list(base_model["path"].glob("*.npy"))

    for base_file in tqdm(all_base_files, desc="Processing files"):
        # 阶段1: 收集所有数据计算全局范围
        all_motion_values = []

        # 处理GT数据
        gt_file = gt_path / base_file.name
        seq_gt = np.load(gt_file)
        motion_gt = np.linalg.norm(seq_gt[1:] - seq_gt[:-1], axis=2)
        all_motion_values.extend([motion_gt.mean(0), motion_gt.std(0)])

        # 处理每个模型的预测数据（非STtb模型需要添加test_前缀）
        for model in model_configs:
            # 构造预测文件名：非STtb模型添加test_前缀
            pred_name = f"test_{base_file.name}" if model["name"] != "STtb" else base_file.name
            pred_file = model["path"] / pred_name

            # 加载预测数据并计算运动向量
            seq_pred = np.load(pred_file)
            motion_pred = np.linalg.norm(seq_pred[1:] - seq_pred[:-1], axis=2)
            all_motion_values.extend([motion_pred.mean(0), motion_pred.std(0)])

        # 计算全局范围
        global_max = max([arr.max() for arr in all_motion_values])
        global_min = min([arr.min() for arr in all_motion_values])
        scale_factor, unit_label = calculate_scale_factor(global_max)

        # 阶段2: 统一渲染所有热力图
        # 渲染GT
        _render_and_save(template_all, motion_gt.mean(0) / scale_factor,
                         base_file.stem, "gt", "mean",
                         global_min / scale_factor, global_max / scale_factor, unit_label)
        _render_and_save(template_all, motion_gt.std(0) / scale_factor,
                         base_file.stem, "gt", "std",
                         global_min / scale_factor, global_max / scale_factor, unit_label)

        # 渲染每个模型
        for model in model_configs:
            pred_name = f"test_{base_file.name}" if model["name"] != "STtb" else base_file.name
            pred_file = model["path"] / pred_name

            seq_pred = np.load(pred_file)
            motion_pred = np.linalg.norm(seq_pred[1:] - seq_pred[:-1], axis=2)

            _render_and_save(template_all, motion_pred.mean(0) / scale_factor,
                             base_file.stem, model["name"], "mean",
                             global_min / scale_factor, global_max / scale_factor, unit_label)
            _render_and_save(template_all, motion_pred.std(0) / scale_factor,
                             base_file.stem, model["name"], "std",
                             global_min / scale_factor, global_max / scale_factor, unit_label)

        gc.collect()


def _render_and_save(template, data, base_name, model_name, metric_type, vmin, vmax, unit):
    """统一渲染和保存的辅助函数"""
    output_name = f"{base_name}_{model_name}_{metric_type}.png"
    render_heatmap(
        template_all=template,
        motion_vec=data,
        output_path=str(output_path / output_name),
        global_min=vmin,
        global_max=vmax,
        unit_label=unit
    )


def calculate_scale_factor(max_val):
    """根据最大值自动确定合适的科学记数法单位和缩放因子"""
    if max_val >= 1:
        return 1, "mm"
    # 计算指数
    exponent = int(np.floor(np.log10(max_val)))
    scale_factor = 10 ** exponent
    # 构建上标格式
    superscript_map = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    exponent_str = str(abs(exponent)).translate(superscript_map)
    unit_label = f"×10⁻{exponent_str} mm" if exponent < 0 else "mm"
    return scale_factor, unit_label


def render_heatmap(template_all, motion_vec, output_path, global_min, global_max, unit_label):
    # 修改归一化逻辑（使用统一范围）
    norm_motion_vec = (motion_vec - global_min) / (global_max - global_min + 1e-8)
    colors = blue_red_cmap(norm_motion_vec)[:, :3]

    # 将颜色应用到网格
    mesh = trimesh.Trimesh(vertices=template_all, faces=trimesh.load(reference_path).faces, vertex_colors=colors)

    # 渲染设置
    py_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(py_mesh)
    scene.add(cam, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    # 渲染并保存主图像
    color, _ = r.render(scene)

    # 生成高分辨率统一颜色条并合并到图像右下角
    colorbar_img = generate_colorbar(global_min, global_max, color.shape[0] // 2, unit_label)
    combined_img = add_colorbar_to_image(color, colorbar_img)

    # 保存最终图像
    cv2.imwrite(output_path, combined_img)

    del mesh
    gc.collect()


def generate_colorbar(min_val, max_val, height, unit_label):
    fig, ax = plt.subplots(figsize=(2, height / 100), dpi=100)  # 调整dpi和尺寸
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    # 颜色映射
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=blue_red_cmap), ax=ax)
    cb.ax.set_title(f"({unit_label})", fontsize=12)
    cb.ax.tick_params(labelsize=12)

    # 移除背景和边框
    ax.remove()
    fig.subplots_adjust(left=0.3, right=0.7, top=0.9, bottom=0.1)

    # 渲染为图像
    fig.canvas.draw()
    colorbar_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    colorbar_image = colorbar_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)  # 关闭图形，释放内存
    return colorbar_image


def add_colorbar_to_image(main_image, colorbar_image):
    h, w = main_image.shape[:2]
    ch, cw = colorbar_image.shape[:2]

    # 定位颜色条的位置，放在右下角
    combined_image = main_image.copy()
    combined_image[h - ch - 10:h - 10, w - cw - 30:w - 30] = colorbar_image
    return combined_image


if __name__ == '__main__':
    generate_heatmap()
