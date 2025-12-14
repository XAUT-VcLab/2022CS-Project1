import torch
import numpy as np
from tqdm import tqdm
from FLAME.FLAME import FLAME
from DataProcess.Dataload import get_dataloader


class Config:
    def __init__(self):
        # 模型文件路径 male/female/generic
        self.flame_model_path = '/home/chensheng/1Project/Project1/FLAME2020/flame_model/generic_model.pkl'
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
        self.static_landmark_embedding_path = ('/home/chensheng/1Project/Project1/FLAME2020/flame_model'
                                               '/flame_static_embedding.pkl')
        # 动态标志嵌入路径
        self.dynamic_landmark_embedding_path = ('/home/chensheng/1Project/Project1/FLAME2020/flame_model'
                                                '/flame_dynamic_embedding.npy')


def generate_and_save_mean_template():
    """
    生成平均姿态模板并保存。
    参数:
        flame_model: 预加载的FLAME模型，用于生成顶点数据。
        dataloader: 数据加载器，用于遍历所有数据样本。
        device: 计算设备（CPU或GPU）。
        save_path: 保存平均姿态模板的路径。
    """
    dataloader = get_dataloader('/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/mead/mead',
                                batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flame_model = FLAME(Config()).to(device)
    vertex_sum = None  # 用于累积顶点和
    n_total = 0  # 有效顶点数累积
    with torch.no_grad():
        for _, _, _, shape, exp, jaw, mask in tqdm(dataloader, desc="生成平均姿态模板"):
            # 数据预处理：仅保留有效数据部分
            shape, exp, jaw, n = shape.to(device), exp.to(device), jaw.to(device), int(mask.to(device).sum())
            pose_params = torch.cat((torch.zeros((256, 3)).to(device), jaw.squeeze(0)), dim=1)

            # 使用FLAME模型生成零姿态顶点
            vertices, _ = flame_model(shape.squeeze(0), exp.squeeze(0), pose_params)
            vertices = vertices[:n, :, :]  # 只取有效帧

            # 累加顶点和，并更新有效顶点数量
            if vertex_sum is None:
                vertex_sum = vertices.sum(dim=0)  # 初始化累加和
            else:
                vertex_sum += vertices.sum(dim=0)
            n_total += n  # 更新总帧数

    # 计算平均顶点，保存为 .npy 文件
    save_path = "/home/chensheng/1Project/Project1/FLAME2020/mean_template.npy"
    mean_template = (vertex_sum / n_total).cpu().numpy()
    np.save(save_path, mean_template)
    print(f"平均姿态模板已保存至 {save_path}")


generate_and_save_mean_template()
