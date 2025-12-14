import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from FLAME.FLAME import FLAME
from DataProcess.Dataload import get_dataloader


class Config:
    def __init__(self):
        # 模型文件路径 male/female/generic/flame2023/flame2023_no_jaw
        self.flame_model_path = '/home/chensheng/1Project/Project1/FLAME/flame_model/generic_model.pkl'
        # 批处理大小
        self.batch_size = 256
        # 是否使用面部轮廓
        self.use_face_contour = True
        # 形状参数数量
        self.shape_params = 300
        # 表情参数数量
        self.expression_params = 100
        # 是否使用 3D 位移
        self.use_3D_translation = True
        # 静态标志嵌入路径
        self.static_landmark_embedding_path = '/home/chensheng/1Project/Project1/FLAME/flame_model/flame_static_embedding.pkl'
        # 动态标志嵌入路径
        self.dynamic_landmark_embedding_path = '/home/chensheng/1Project/Project1/FLAME/flame_model/flame_dynamic_embedding.npy'


def predict_model():
    # 设置保存路径
    gt_save_path = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/mead/mead_vertices"
    #audio_save_path = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/mead/mead_audio"
    os.makedirs(gt_save_path, exist_ok=True)
    #os.makedirs(audio_save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flame_model = FLAME(Config()).to(device)
    dataloader = get_dataloader("/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/mead/mead",
                                batch_size=1)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, (video_token, _, text, audio, _, exp, jaw, mask) in pbar:
            # 数据预处理
            audio, exp, jaw, n = (audio.to(device), exp.to(device), jaw.to(device),
                                  int(mask.to(device).sum()))
            # 保存音频到文件
            #audio_file_path = os.path.join(audio_save_path, f"{video_token[0]}.wav")
            #audio_data = audio[:, :n * 1920, :].squeeze().cpu().numpy()
            #wavfile.write(audio_file_path, 48000, audio_data.astype(np.float32))

            # 原始顶点，用于评估基准
            shape_params = torch.zeros((256, 300)).to(device)
            pose_params = torch.cat((torch.zeros((256, 3)).to(device), jaw.squeeze(0)), dim=1)
            vertices_gt, _ = flame_model(shape_params, exp.squeeze(0), pose_params)
            vertices_gt = vertices_gt[:n, :, :].cpu().numpy()
            # 保存原始顶点到文件
            gt_file_path = os.path.join(gt_save_path, f"{video_token[0]}.npy")
            np.save(gt_file_path, vertices_gt)


# 执行预测
if __name__ == "__main__":
    predict_model()
