import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import h5py
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from DataProcess.TA_MEAD.text_gen import generate_text_for_video


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.files = self.get_all_files(data_dir)
        self.au_value_df = pd.read_csv("/home/chensheng/1Project/Project1/DataProcess/TA_MEAD/AU_value.csv")
        with open("/home/chensheng/1Project/Project1/DataProcess/TA_MEAD/complex_emotion.yaml", 'r') as f:
            self.emotion_table = yaml.full_load(f)
        self.au_description_df = pd.read_csv("/home/chensheng/1Project/Project1/DataProcess/TA_MEAD/ActionUnit.csv")
        self.intensity_df = pd.read_csv("/home/chensheng/1Project/Project1/DataProcess/TA_MEAD/Intensity.csv")
        self.au_intensity_split_df = pd.read_csv("/home/chensheng/1Project/Project1/DataProcess/TA_MEAD"
                                                 "/AU_intensity_split.csv")

        self.person_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 'M022', 'M023', 'M024',
                           'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035',
                           'M037', 'M039', 'M040', 'M041', 'M042', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018',
                           'W019', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029', 'W033', 'W035', 'W036', 'W037',
                           'W038', 'W040']
        self.emotion_dict = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgusted', 'angry', 'contempt']
        self.intensity_dict = ['level1', 'level2', 'level3']
        self.person_id_to_one_hot = {pid: self.create_one_hot(len(self.person_ids), idx) for idx, pid in
                                     enumerate(self.person_ids)}
        self.emotion_to_one_hot = {pid: self.create_one_hot(len(self.emotion_dict), idx) for idx, pid in
                                   enumerate(self.emotion_dict)}
        self.intensity_to_one_hot = {pid: self.create_one_hot(len(self.intensity_dict), idx) for idx, pid in
                                     enumerate(self.intensity_dict)}

    def get_all_files(self, directory):
        """ 获取指定目录下的所有文件名 """
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.hdf5'):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
        print(f"Number of files used: {len(all_files)}")
        return all_files

    def create_one_hot(self, length, index):
        """ 创建长度为 length 的 one-hot 编码，位置 index 为 1 """
        one_hot = torch.zeros(length)
        one_hot[index] = 1
        return one_hot

    def extract_video_token(self, file_path):
        """ 从文件路径中提取标识符 """
        file_name = os.path.basename(file_path)
        base_name = file_name.replace('.hdf5', '')
        return base_name

    def padding_sequence_length(self, data, interval):
        """对较短的序列进行填充，对过长的序列进行截断"""
        seq_len, feature_dim = data.shape
        # 计算需要保留的最大倍数
        N = seq_len // interval
        max_blocks = 8  # 可以调整的最大块数
        new_seq_len = max_blocks * interval
        if N >= max_blocks:
            # 截断数据，只保留前 32 * interval 长度
            padded_data = data[:new_seq_len, :]
            mask = torch.ones(new_seq_len, dtype=torch.float32)
        else:
            # 填充数据到 N * interval 长度，如果长度不足则进行填充
            padded_data = torch.zeros((new_seq_len, feature_dim), dtype=data.dtype)  # 使用0填充
            padded_data[:N * interval, :] = data[:N * interval, :]  # 原数据填入
            # 创建掩码，实际数据部分为 1，填充部分为 0
            mask = torch.zeros(new_seq_len, dtype=torch.float32)
            mask[:N * interval] = 1  # 原始数据部分标记为 1
        return padded_data, mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        video_token = self.extract_video_token(file_path)  # 使用文件路径
        # 提取person_id并查找对应的one-hot编码
        person_id = video_token.split('_')[0]
        emotion = video_token.split('_')[2]
        intensity = video_token.split('_')[3]
        person_one_hot = self.person_id_to_one_hot.get(person_id, torch.zeros(len(self.person_ids)))
        emotion_one_hot = self.emotion_to_one_hot.get(emotion, torch.zeros(len(self.emotion_dict)))
        intensity_one_hot = self.intensity_to_one_hot.get(intensity, torch.zeros(len(self.intensity_dict)))
        emotion_one_hot = torch.cat((emotion_one_hot, intensity_one_hot), dim=0)
        # 从文件中加载
        with h5py.File(file_path, 'r') as f:
            audio = torch.tensor(f['audio'][:])
            shape_data = torch.tensor(f['shape'][:])
            exp_data = torch.tensor(f['exp'][:])
            jaw_data = torch.tensor(f['jaw'][:])

        # 分割 data 到指定长度
        audio, _ = self.padding_sequence_length(audio.permute(1, 0), 61440)
        shape_data, mask = self.padding_sequence_length(shape_data, 32)
        exp_data, _ = self.padding_sequence_length(exp_data, 32)
        jaw_data, _ = self.padding_sequence_length(jaw_data, 32)
        return (
            video_token,
            person_one_hot,
            emotion_one_hot,
            generate_text_for_video(
                video_token,
                self.au_value_df,
                self.emotion_table,
                self.au_description_df,
                self.intensity_df,
                use_intensity=True,
                AU_intensity_split_df=self.au_intensity_split_df,
                use_emotion=True,
                use_AU=False
            ), audio, shape_data, exp_data, jaw_data, mask
        )


def get_dataloader(data_dir, batch_size=1):
    dataset = CustomDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=40, pin_memory=True)
    return dataloader


# 测试部分
if __name__ == "__main__":
    # 配置路径
    data_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/test"
    # 创建数据加载器
    dataloader = get_dataloader(data_dir, batch_size=2)
    # 获取第一个数据
    for video_token, person_one_hot, emotion_one_hot, text, audio, shape_data, exp_data, jaw_data, mask in dataloader:
        print("video token:", video_token)
        print("person_one_hot.shape", person_one_hot.shape)
        print("emotion_one_hot.shape", emotion_one_hot.shape)
        print("text:", text)
        print("audio.shape", audio.shape)
        print("shape_data.shape", shape_data.shape)
        print("exp_data.shape", exp_data.shape)
        print(exp_data)
        print("jaw_data.shape", jaw_data.shape)
        print(jaw_data)
        print("mask.shape", mask.shape)
        print("mask_sum", int(mask.sum()))
        break
