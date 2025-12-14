import os
import cv2
import ffmpeg
import trimesh
import pyrender
import numpy as np
from tqdm import tqdm
from pathlib import Path
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 文件路径
#audio_path = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/test_audio"
audio_path = "/home/chensheng/2Model/CodeTalker/mead/wav"
#vertices_path = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/test_vertices"
#vertices_path = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/STtb_vertices"
#vertices_path = "/home/chensheng/2Model/CodeTalker/RUN/mead/CodeTalker_s2_2/result/npy"
#vertices_path = "/home/chensheng/2Model/FaceDiffuser/result"
#vertices_path = "/home/chensheng/2Model/FaceFormer/mead/result"
vertices_path = "/home/chensheng/2Model/ProbTalk3D/results/evaluation/vqvae_pred/2fiqeonq/0.2_multi/vert"
output_video_path = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/ProbTalk3D_video"

# 读取文件列表
vertices_files = sorted(Path(vertices_path).glob("*.npy"))
# 创建输出文件夹
os.makedirs(output_video_path, exist_ok=True)
# 加载模板网格
template_path = "/home/chensheng/1Project/Project1/FLAME/flame_sample.ply"
ref_mesh = trimesh.load_mesh(template_path)

# 相机和渲染设置
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
cam = pyrender.PerspectiveCamera(yfov=np.pi / 20, aspectRatio=1.414)
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0]])
r = pyrender.OffscreenRenderer(960, 760)


def create_video(vertices_files, audio_path, output_folder):
    for data_file in tqdm(vertices_files, desc="Processing files"):
        file_name = data_file.stem
        video_output_path = Path(output_folder) / f"{file_name}_without_audio.mp4"
        video = cv2.VideoWriter(str(video_output_path), fourcc, 25, (960, 760))

        # 加载顶点数据
        motion_data = np.load(data_file)
        motion_data = motion_data.reshape(-1, ref_mesh.vertices.shape[0], 3)

        for frame_data in motion_data:
            ref_mesh.vertices = frame_data
            py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
            scene = pyrender.Scene()
            scene.add(py_mesh)
            scene.add(cam, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            color, _ = r.render(scene)
            video.write(color)
        video.release()

        # 添加音频
        audio_file = Path(audio_path) / f"{file_name}.wav"
        input_video = ffmpeg.input(str(video_output_path))
        input_audio = ffmpeg.input(str(audio_file))
        output_with_audio = Path(output_folder) / f"{file_name}.mp4"
        (ffmpeg.concat(input_video, input_audio, v=1, a=1).output(str(output_with_audio)).
         run(overwrite_output=True, quiet=True))
        video_output_path.unlink()  # 删除无音频版本


# 生成原始和预测视频
create_video(vertices_files, audio_path, output_video_path)
