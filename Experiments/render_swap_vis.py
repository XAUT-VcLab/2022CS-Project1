# Experiments/render_swap_vis.py
# python Experiments/render_swap_vis.py --stage all --pair_type all

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import json
import yaml
import argparse
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import cv2
import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

# 项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataProcess.Dataload import CustomDataset
from FLAME.FLAME import FLAME


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")
    default_stage1_root = os.path.join(os.path.dirname(__file__), "stage1_swap_results")
    default_stage2_root = os.path.join(os.path.dirname(__file__), "stage2_swap_results")
    default_save_root = os.path.join(os.path.dirname(__file__), "swap_vis")

    parser = argparse.ArgumentParser(description="Render swap experiment visualizations for DecTalk3D.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    parser.add_argument("--stage", type=str, default="stage2", choices=["stage1", "stage2", "all"])
    parser.add_argument(
        "--pair_type",
        type=str,
        default="text_emotion",
        choices=["text_emotion", "text_intensity", "identity", "all"],
    )
    parser.add_argument("--group_name", type=str, default=None, help="Render a specific group_xxxxx")
    parser.add_argument("--max_groups", type=int, default=-1, help="-1 means all groups")
    parser.add_argument("--gpu", type=int, default=None)

    parser.add_argument("--stage1_root", type=str, default=default_stage1_root)
    parser.add_argument("--stage2_root", type=str, default=default_stage2_root)
    parser.add_argument("--save_root", type=str, default=default_save_root)

    parser.add_argument("--num_frames", type=int, default=4, help="How many frames to sample uniformly")

    # 内部高分辨率渲染
    parser.add_argument("--render_width", type=int, default=1400, help="Internal render width")
    parser.add_argument("--render_height", type=int, default=1000, help="Internal render height")

    # 最终输出格子尺寸
    parser.add_argument("--cell_width", type=int, default=220, help="Output cell width")
    parser.add_argument("--cell_height", type=int, default=360, help="Output cell height")

    # 更紧凑布局
    parser.add_argument("--row_gap", type=int, default=2, help="Vertical gap between rows")
    parser.add_argument("--col_gap", type=int, default=2, help="Horizontal gap between columns")
    parser.add_argument("--title_height", type=int, default=34, help="Top title area height")
    parser.add_argument("--meta_height", type=int, default=34, help="Source/target meta area height")
    parser.add_argument("--col_header_height", type=int, default=26, help="Column header area height")

    parser.add_argument("--crop_padding", type=int, default=6, help="Padding after tight crop")
    parser.add_argument("--white_threshold", type=int, default=245, help="Background threshold for auto crop")
    parser.add_argument("--font_path", type=str, default=None, help="Optional font path.")

    return parser.parse_args()


def resolve_path_from_config_or_args(config: Dict, arg_value: str, config_key: str, name: str) -> str:
    candidates = [arg_value, config.get(config_key, None)]
    for p in candidates:
        if p is not None and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {name}. Tried: {candidates}")


def ensure_tensor(x, device: torch.device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def infer_mask_from_gt(exp_gt: np.ndarray, jaw_gt: np.ndarray) -> np.ndarray:
    exp_abs = np.abs(exp_gt).sum(axis=-1)   # [1, T]
    jaw_abs = np.abs(jaw_gt).sum(axis=-1)   # [1, T]
    valid = (exp_abs + jaw_abs) > 0
    return valid.astype(np.float32)[0]


def mask_to_valid_length(mask: np.ndarray) -> int:
    if mask.ndim != 1:
        mask = mask.reshape(-1)
    valid = mask > 0
    if valid.sum() == 0:
        return len(mask)
    return int(np.where(valid)[0][-1] + 1)


def crop_sequence_to_valid_length(exp_arr: np.ndarray, jaw_arr: np.ndarray, valid_len: int):
    return exp_arr[:, :valid_len, :], jaw_arr[:, :valid_len, :]


def get_frame_indices(seq_len: int, num_frames: int) -> List[int]:
    if seq_len <= 0:
        return [0]
    if num_frames <= 1:
        return [seq_len // 2]

    idxs = np.linspace(0, seq_len - 1, num=num_frames)
    idxs = np.round(idxs).astype(int).tolist()
    idxs = sorted(list(dict.fromkeys(idxs)))
    return idxs


def build_dataset_and_index(test_dir: str):
    dataset = CustomDataset(test_dir)
    token_to_index = {}
    for idx, file_path in enumerate(dataset.files):
        token = dataset.extract_video_token(file_path)
        if token not in token_to_index:
            token_to_index[token] = idx
    return dataset, token_to_index


def load_shape_from_token(
    dataset: CustomDataset,
    token_to_index: Dict[str, int],
    video_token: str,
) -> np.ndarray:
    if video_token not in token_to_index:
        raise KeyError(f"video_token not found in dataset: {video_token}")

    idx = token_to_index[video_token]
    _, _, _, _, _, shape_data, _, _, _ = dataset[idx]

    if isinstance(shape_data, torch.Tensor):
        shape_data = shape_data.detach().cpu().numpy()
    shape_data = np.asarray(shape_data, dtype=np.float32)

    if shape_data.ndim == 1:
        shape_data = shape_data[None, :]
    elif shape_data.ndim == 2 and shape_data.shape[0] != 1:
        shape_data = shape_data[:1]

    return shape_data


class FlameDecoderCache:
    def __init__(
        self,
        flame_model_path: str,
        static_landmark_embedding_path: str,
        dynamic_landmark_embedding_path: str,
        device: torch.device
    ):
        self.flame_model_path = flame_model_path
        self.static_landmark_embedding_path = static_landmark_embedding_path
        self.dynamic_landmark_embedding_path = dynamic_landmark_embedding_path
        self.device = device
        self.cache = {}

    def get(self, shape_dim: int, exp_dim: int):
        key = (shape_dim, exp_dim)
        if key in self.cache:
            return self.cache[key]

        cfg = SimpleNamespace(
            flame_model_path=self.flame_model_path,
            static_landmark_embedding_path=self.static_landmark_embedding_path,
            dynamic_landmark_embedding_path=self.dynamic_landmark_embedding_path,
            batch_size=1,
            use_face_contour=False,
            shape_params=int(shape_dim),
            expression_params=int(exp_dim),
            use_3D_translation=False,
        )
        flame = FLAME(cfg).to(self.device)
        flame.eval()
        self.cache[key] = flame
        return flame


def decode_vertices_sequence(
    shape_arr: np.ndarray,
    exp_arr: np.ndarray,
    jaw_arr: np.ndarray,
    flame_cache: FlameDecoderCache,
    device: torch.device,
) -> np.ndarray:
    shape_arr = np.asarray(shape_arr, dtype=np.float32)
    exp_arr = np.asarray(exp_arr, dtype=np.float32)
    jaw_arr = np.asarray(jaw_arr, dtype=np.float32)

    T = exp_arr.shape[1]
    shape_dim = shape_arr.shape[-1]
    exp_dim = exp_arr.shape[-1]

    flame = flame_cache.get(shape_dim, exp_dim)

    vertices_list = []
    with torch.no_grad():
        shape_t = ensure_tensor(shape_arr, device)

        for t in range(T):
            exp_t = ensure_tensor(exp_arr[:, t, :], device)
            jaw_t = ensure_tensor(jaw_arr[:, t, :], device)

            pose_t = torch.zeros((1, 6), dtype=torch.float32, device=device)
            pose_t[:, 3:] = jaw_t

            vertices, _ = flame(
                shape_params=shape_t,
                expression_params=exp_t,
                pose_params=pose_t,
            )
            vertices_list.append(vertices[0].detach().cpu().numpy())

    return np.stack(vertices_list, axis=0)


class MeshRenderer:
    def __init__(self, template_path: str, width: int, height: int):
        self.template_mesh = trimesh.load_mesh(template_path)
        self.faces = self.template_mesh.faces.copy()
        self.width = width
        self.height = height

        self.cam = pyrender.PerspectiveCamera(yfov=np.pi / 20, aspectRatio=1.414)
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        self.camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def render_frame(self, vertices: np.ndarray) -> np.ndarray:
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        py_mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=[0.3, 0.3, 0.3])
        scene.add(py_mesh)
        scene.add(self.cam, pose=self.camera_pose)
        scene.add(self.light, pose=self.camera_pose)

        color, _ = self.renderer.render(scene)
        return color

    def close(self):
        self.renderer.delete()


def find_nonwhite_bbox(img: np.ndarray, white_threshold: int):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray < white_threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, img.shape[0], 0, img.shape[1]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(y_min), int(y_max + 1), int(x_min), int(x_max + 1)


def tight_crop(img: np.ndarray, bbox, pad: int):
    y0, y1, x0, x1 = bbox
    h, w = img.shape[:2]
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    return img[y0:y1, x0:x1]


def resize_keep_center(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def get_font(font_path: str = None, size: int = 18):
    candidates = [
        font_path,
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if p is not None and os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def draw_center_text(canvas: np.ndarray, text: str, y0: int, h: int, font, color=(0, 0, 0)):
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (canvas.shape[1] - tw) // 2
    y = y0 + (h - th) // 2
    draw.text((x, y), text, fill=color, font=font)
    return np.array(pil)


def draw_left_text(canvas: np.ndarray, text: str, y0: int, h: int, font, color=(0, 0, 0), x_pad=8):
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    y = y0 + (h - th) // 2
    draw.text((x_pad, y), text, fill=color, font=font)
    return np.array(pil)


def make_grid(
    rows: List[List[np.ndarray]],
    row_gap: int,
    col_gap: int,
    title: str,
    meta_left: str,
    meta_right: str,
    col_titles: List[str],
    title_height: int,
    meta_height: int,
    col_header_height: int,
    font_path: str = None,
) -> np.ndarray:
    num_rows = len(rows)
    num_cols = len(rows[0])

    cell_h, cell_w = rows[0][0].shape[:2]

    title_block = title_height
    meta_block = meta_height
    col_block = col_header_height

    total_h = title_block + meta_block + col_block + num_rows * cell_h + (num_rows - 1) * row_gap
    total_w = num_cols * cell_w + (num_cols - 1) * col_gap

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    font_title = get_font(font_path, size=18)
    font_meta = get_font(font_path, size=14)
    font_col = get_font(font_path, size=14)

    canvas = draw_center_text(canvas, title, 0, title_block, font_title, color=(0, 0, 0))
    canvas = draw_left_text(canvas, meta_left, title_block, meta_block, font_meta, color=(40, 40, 40), x_pad=8)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), meta_right, font=font_meta)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = total_w - tw - 8
    y = title_block + (meta_block - th) // 2
    draw.text((x, y), meta_right, fill=(40, 40, 40), font=font_meta)
    canvas = np.array(pil)

    y_col = title_block + meta_block
    for c in range(num_cols):
        x0 = c * (cell_w + col_gap)
        sub = canvas[:, x0:x0+cell_w]
        sub = draw_center_text(sub, col_titles[c], y_col, col_block, font_col, color=(0, 0, 0))
        canvas[:, x0:x0+cell_w] = sub

    y0 = title_block + meta_block + col_block
    for r in range(num_rows):
        for c in range(num_cols):
            x = c * (cell_w + col_gap)
            y = y0 + r * (cell_h + row_gap)
            canvas[y:y+cell_h, x:x+cell_w] = rows[r][c]

    return canvas


def get_result_paths(stage: str, group_dir: str):
    if stage == "stage1":
        return {
            "source_orig_exp": os.path.join(group_dir, "source_rec_exp.npy"),
            "source_orig_jaw": os.path.join(group_dir, "source_rec_jaw.npy"),
            "target_orig_exp": os.path.join(group_dir, "target_rec_exp.npy"),
            "target_orig_jaw": os.path.join(group_dir, "target_rec_jaw.npy"),
        }
    elif stage == "stage2":
        return {
            "source_orig_exp": os.path.join(group_dir, "source_gen_exp.npy"),
            "source_orig_jaw": os.path.join(group_dir, "source_gen_jaw.npy"),
            "target_orig_exp": os.path.join(group_dir, "target_gen_exp.npy"),
            "target_orig_jaw": os.path.join(group_dir, "target_gen_jaw.npy"),
        }
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def render_one_group(
    stage: str,
    pair_type: str,
    group_dir: str,
    dataset: CustomDataset,
    token_to_index: Dict[str, int],
    flame_cache: FlameDecoderCache,
    mesh_renderer: MeshRenderer,
    device: torch.device,
    save_dir: str,
    args,
):
    meta = load_json(os.path.join(group_dir, "meta.json"))
    result_paths = get_result_paths(stage, group_dir)

    source_token = meta["source_video_token"]
    target_token = meta["target_video_token"]

    source_shape = load_shape_from_token(dataset, token_to_index, source_token)
    target_shape = load_shape_from_token(dataset, token_to_index, target_token)

    source_gt_exp = load_npy(os.path.join(group_dir, "source_gt_exp.npy"))
    source_gt_jaw = load_npy(os.path.join(group_dir, "source_gt_jaw.npy"))
    target_gt_exp = load_npy(os.path.join(group_dir, "target_gt_exp.npy"))
    target_gt_jaw = load_npy(os.path.join(group_dir, "target_gt_jaw.npy"))

    source_orig_exp = load_npy(result_paths["source_orig_exp"])
    source_orig_jaw = load_npy(result_paths["source_orig_jaw"])
    target_orig_exp = load_npy(result_paths["target_orig_exp"])
    target_orig_jaw = load_npy(result_paths["target_orig_jaw"])

    source_swap_exp = load_npy(os.path.join(group_dir, "source_swap_exp.npy"))
    source_swap_jaw = load_npy(os.path.join(group_dir, "source_swap_jaw.npy"))
    target_swap_exp = load_npy(os.path.join(group_dir, "target_swap_exp.npy"))
    target_swap_jaw = load_npy(os.path.join(group_dir, "target_swap_jaw.npy"))

    source_valid_len = mask_to_valid_length(infer_mask_from_gt(source_gt_exp, source_gt_jaw))
    target_valid_len = mask_to_valid_length(infer_mask_from_gt(target_gt_exp, target_gt_jaw))

    source_orig_exp, source_orig_jaw = crop_sequence_to_valid_length(source_orig_exp, source_orig_jaw, source_valid_len)
    source_swap_exp, source_swap_jaw = crop_sequence_to_valid_length(source_swap_exp, source_swap_jaw, source_valid_len)

    target_orig_exp, target_orig_jaw = crop_sequence_to_valid_length(target_orig_exp, target_orig_jaw, target_valid_len)
    target_swap_exp, target_swap_jaw = crop_sequence_to_valid_length(target_swap_exp, target_swap_jaw, target_valid_len)

    source_orig_vertices = decode_vertices_sequence(source_shape, source_orig_exp, source_orig_jaw, flame_cache, device)
    target_orig_vertices = decode_vertices_sequence(target_shape, target_orig_exp, target_orig_jaw, flame_cache, device)
    source_swap_vertices = decode_vertices_sequence(source_shape, source_swap_exp, source_swap_jaw, flame_cache, device)
    target_swap_vertices = decode_vertices_sequence(target_shape, target_swap_exp, target_swap_jaw, flame_cache, device)

    source_frame_indices = get_frame_indices(len(source_orig_vertices), args.num_frames)
    target_frame_indices = get_frame_indices(len(target_orig_vertices), args.num_frames)

    source_bbox = None
    target_bbox = None

    source_probe = mesh_renderer.render_frame(source_orig_vertices[source_frame_indices[0]])
    source_bbox = find_nonwhite_bbox(source_probe, args.white_threshold)

    target_probe = mesh_renderer.render_frame(target_orig_vertices[target_frame_indices[0]])
    target_bbox = find_nonwhite_bbox(target_probe, args.white_threshold)

    num_rows = max(len(source_frame_indices), len(target_frame_indices))
    rows = []

    for ridx in range(num_rows):
        if ridx < len(source_frame_indices):
            si = source_frame_indices[ridx]
            img_source_orig = mesh_renderer.render_frame(source_orig_vertices[si])
            img_source_swap = mesh_renderer.render_frame(source_swap_vertices[min(si, len(source_swap_vertices) - 1)])

            img_source_orig = tight_crop(img_source_orig, source_bbox, args.crop_padding)
            img_source_swap = tight_crop(img_source_swap, source_bbox, args.crop_padding)

            img_source_orig = resize_keep_center(img_source_orig, args.cell_width, args.cell_height)
            img_source_swap = resize_keep_center(img_source_swap, args.cell_width, args.cell_height)
        else:
            img_source_orig = np.ones((args.cell_height, args.cell_width, 3), dtype=np.uint8) * 255
            img_source_swap = np.ones((args.cell_height, args.cell_width, 3), dtype=np.uint8) * 255

        if ridx < len(target_frame_indices):
            ti = target_frame_indices[ridx]
            img_target_orig = mesh_renderer.render_frame(target_orig_vertices[ti])
            img_target_swap = mesh_renderer.render_frame(target_swap_vertices[min(ti, len(target_swap_vertices) - 1)])

            img_target_orig = tight_crop(img_target_orig, target_bbox, args.crop_padding)
            img_target_swap = tight_crop(img_target_swap, target_bbox, args.crop_padding)

            img_target_orig = resize_keep_center(img_target_orig, args.cell_width, args.cell_height)
            img_target_swap = resize_keep_center(img_target_swap, args.cell_width, args.cell_height)
        else:
            img_target_orig = np.ones((args.cell_height, args.cell_width, 3), dtype=np.uint8) * 255
            img_target_swap = np.ones((args.cell_height, args.cell_width, 3), dtype=np.uint8) * 255

        rows.append([img_source_orig, img_target_orig, img_source_swap, img_target_swap])

    title = f"{stage} | {pair_type} | {Path(group_dir).name}"
    meta_left = f"Source: {source_token} ({source_valid_len}/{source_gt_exp.shape[1]})"
    meta_right = f"Target: {target_token} ({target_valid_len}/{target_gt_exp.shape[1]})"

    grid = make_grid(
        rows=rows,
        row_gap=args.row_gap,
        col_gap=args.col_gap,
        title=title,
        meta_left=meta_left,
        meta_right=meta_right,
        col_titles=["Source Original", "Target Original", "Source Swapped", "Target Swapped"],
        title_height=args.title_height,
        meta_height=args.meta_height,
        col_header_height=args.col_header_height,
        font_path=args.font_path,
    )

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{Path(group_dir).name}.png")
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[Saved] {out_path}")


def main():
    args = parse_args()
    config = load_yaml(args.config)

    device = torch.device(
        f"cuda:{args.gpu if args.gpu is not None else config['predict']['gpu']}"
        if torch.cuda.is_available() else "cpu"
    )

    test_dir = config["test_file_path"]
    dataset, token_to_index = build_dataset_and_index(test_dir)

    flame_model_path = resolve_path_from_config_or_args(config, None, "flame_model", "flame_model")
    static_lmk_path = resolve_path_from_config_or_args(config, None, "static_landmark_embedding", "static_landmark_embedding")
    dynamic_lmk_path = resolve_path_from_config_or_args(config, None, "dynamic_landmark_embedding", "dynamic_landmark_embedding")
    template_candidates = [
        "/home/chensheng/1Project/Project1/FLAME/flame_sample.ply",
        config.get("template", None),
    ]

    template_path = None
    for p in template_candidates:
        if p is not None and os.path.exists(p):
            template_path = p
            break

    if template_path is None:
        raise FileNotFoundError(f"Cannot find a valid mesh template. Tried: {template_candidates}")

    flame_cache = FlameDecoderCache(
        flame_model_path=flame_model_path,
        static_landmark_embedding_path=static_lmk_path,
        dynamic_landmark_embedding_path=dynamic_lmk_path,
        device=device
    )
    mesh_renderer = MeshRenderer(
        template_path=template_path,
        width=args.render_width,
        height=args.render_height,
    )

    if args.stage == "all":
        stages = ["stage1", "stage2"]
    else:
        stages = [args.stage]

    if args.pair_type == "all":
        pair_types = ["text_emotion", "text_intensity", "identity"]
    else:
        pair_types = [args.pair_type]

    try:
        for stage in stages:
            for pair_type in pair_types:
                result_root = args.stage1_root if stage == "stage1" else args.stage2_root
                result_root = os.path.join(result_root, pair_type)

                if args.group_name is not None:
                    group_dirs = [os.path.join(result_root, args.group_name)]
                else:
                    group_dirs = []
                    if os.path.exists(result_root):
                        for name in sorted(os.listdir(result_root)):
                            full = os.path.join(result_root, name)
                            if os.path.isdir(full) and name.startswith("group_"):
                                group_dirs.append(full)
                    if args.max_groups > 0:
                        group_dirs = group_dirs[:args.max_groups]

                save_dir = os.path.join(args.save_root, stage, pair_type)
                ensure_dir(save_dir)

                print("=" * 80)
                print(f"stage      : {stage}")
                print(f"pair_type  : {pair_type}")
                print(f"result_root: {result_root}")
                print(f"save_dir   : {save_dir}")
                print(f"num_groups : {len(group_dirs)}")
                print("=" * 80)

                for group_dir in group_dirs:
                    if not os.path.exists(group_dir):
                        print(f"[Skip] group dir does not exist: {group_dir}")
                        continue
                    try:
                        render_one_group(
                            stage=stage,
                            pair_type=pair_type,
                            group_dir=group_dir,
                            dataset=dataset,
                            token_to_index=token_to_index,
                            flame_cache=flame_cache,
                            mesh_renderer=mesh_renderer,
                            device=device,
                            save_dir=save_dir,
                            args=args,
                        )
                    except Exception:
                        print(f"[Skip] {group_dir} because of error:")
                        traceback.print_exc()
    finally:
        mesh_renderer.close()

    print("=" * 80)
    print("Render swap visualization finished.")
    print(f"Saved to: {args.save_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()