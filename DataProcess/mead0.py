import io
import os
import csv
import h5py
import hickle as hkl
import subprocess
import torchaudio
import torchaudio.transforms
import torch.nn.functional as F
import random
from tqdm import tqdm

"""
数据集预处理
1、找出文本、音频、人脸共有的数据，再找出音频、人脸时间对齐的数据
2、音频帧数对齐人脸帧数
3、写入音频、shape、exp、jaw数据
4、标识符命名数据
"""

# 设置 ffmpeg 和 ffprobe 的路径
ffmpeg_path = "/root/ffmpeg-7.0.2-amd64-static/ffmpeg"
# 数据集路径
au_value_path = "/home/chensheng/1Project/Project1/DataProcess/TA_MEAD/AU_value.csv"
audio_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/mead/MEAD"
emica_mead_dir = ("/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/mead/processed/reconstructions/EMICA"
                  "-MEAD_flame2020")
emotion_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/mead/processed/emotions/resnet50"
# 保存路径
output_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/mead/mead"


def get_au_value_identifiers(filepath):
    """ 获取AU_value.csv文件中的所有文件标识符 """
    au_identifiers = set()
    with open(filepath, 'r') as csvfile, tqdm(total=os.path.getsize(filepath), desc="Reading AU values") as pbar:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            au_identifiers.add(row[0])
            pbar.update(len(','.join(row)) + 1)
    return au_identifiers


def get_audio_identifiers(directory):
    """ 获取audio目录中的所有文件标识符 """
    audio_identifiers = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.m4a'):
                relative_path = os.path.relpath(root, directory)
                parts = relative_path.split(os.sep)
                person_id = parts[0]
                emotion = parts[2]
                intensity = parts[3].replace('level_', 'level')
                video_id = file.replace('.m4a', '')
                identifier = f"{person_id}_front_{emotion}_{intensity}_{video_id}"
                audio_identifiers.add(identifier)
    return audio_identifiers


def get_emica_mead_identifiers(directory):
    """ 获取EMICA-MEAD_flame2020目录中的所有文件标识符 """
    emica_mead_identifiers = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "shape_pose_cam.hdf5":
                relative_path = os.path.relpath(root, directory)
                parts = relative_path.split(os.sep)
                person_id = parts[0]
                direction = parts[1]
                emotion = parts[2]
                intensity = parts[3].replace('level_', 'level')
                video_id = parts[4]
                identifier = f"{person_id}_{direction}_{emotion}_{intensity}_{video_id}"
                emica_mead_identifiers.add(identifier)
    return emica_mead_identifiers


def load_audio_m4a(file_path):
    """使用 ffmpeg 读取 m4a 文件并返回音频数据"""
    command = [
        ffmpeg_path,
        '-i', file_path,
        '-f', 'wav',
        '-'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_message = result.stderr.decode('utf-8')
        raise RuntimeError(f"ffmpeg error: {error_message}")
    return io.BytesIO(result.stdout)


def calculate_ratio(identifier):
    person_id, direction, emotion, intensity, video_id = identifier.split('_', 4)

    # 音频文件路径
    audio_file = os.path.join(audio_dir, person_id, "audio", emotion, intensity.replace('level', 'level_'),
                              f"{video_id}.m4a")
    # 人脸特征文件路径
    emica_mead_file = os.path.join(emica_mead_dir, person_id, direction, emotion, intensity.replace('level', 'level_'),
                                   video_id, "shape_pose_cam.hdf5")

    with h5py.File(emica_mead_file, 'r') as infile:
        exp_data = infile['exp'][:]
        num_frames_exp = exp_data.shape[1]

        try:
            audio_data = load_audio_m4a(audio_file)
        except RuntimeError as e:
            print(f"Error loading audio for {identifier}: {e}")
            return None

        waveform, sample_rate = torchaudio.load(audio_data)
        audio_duration = waveform.shape[1] / sample_rate

        ratio = audio_duration / num_frames_exp
        return ratio


# 获取各目录中的文件标识符
au_value_identifiers = get_au_value_identifiers(au_value_path)
audio_identifiers = get_audio_identifiers(audio_dir)
emica_mead_identifiers = get_emica_mead_identifiers(emica_mead_dir)

# 找到三者共有的标识符
common_identifiers = au_value_identifiers & audio_identifiers & emica_mead_identifiers
print(f"Identifiers common in all three: {len(common_identifiers)}")

# 随机打乱标识符列表
common_identifiers = list(common_identifiers)
random.shuffle(common_identifiers)

# 计算比值并过滤
filtered_identifiers = []
'''
filtered_identifiers = common_identifiers
'''
for identifier in tqdm(common_identifiers, desc="Calculating ratios"):
    ratio = calculate_ratio(identifier)
    if ratio is not None and 0.038 <= ratio <= 0.040:  # 比值1/25
        filtered_identifiers.append(identifier)

print(f"Filtered identifiers count: {len(filtered_identifiers)}")


# 处理并保存数据
def process_and_save(identifier, output_subdir, error_files):
    person_id, direction, emotion, intensity, video_id = identifier.split('_', 4)

    # 音频文件路径
    audio_file = os.path.join(audio_dir, person_id, "audio", emotion, intensity.replace('level', 'level_'),
                              f"{video_id}.m4a")
    # 人脸特征文件路径
    emica_mead_file = os.path.join(emica_mead_dir, person_id, direction, emotion, intensity.replace('level', 'level_'),
                                   video_id, "shape_pose_cam.hdf5")

    emotion_file = os.path.join(emotion_dir, person_id, direction, emotion, intensity.replace('level', 'level_'),
                                video_id, "emotions.pkl")

    with h5py.File(emica_mead_file, 'r') as infile:
        shape_data = infile['shape'][:]
        exp_data = infile['exp'][:]
        jaw_data = infile['jaw'][:]
        num_frames_exp = shape_data.shape[1]

        # 加载音频数据
        try:
            audio_data = load_audio_m4a(audio_file)
        except RuntimeError as e:
            error_files.append((identifier, str(e)))
            return False

        waveform, sample_rate = torchaudio.load(audio_data)
        waveform = waveform.mean(dim=0, keepdim=True).unsqueeze(0)
        waveform = F.interpolate(waveform, size=(num_frames_exp * 1920,), mode='linear', align_corners=False).squeeze(0)

        with open(emotion_file, 'rb') as ef:
            emotions_data = hkl.load(ef)
            expression_data = emotions_data['expression']

        # 保存到 HDF5 文件
        output_file = os.path.join(output_subdir, f"{identifier}.hdf5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('audio', data=waveform)
            f.create_dataset('emotions', data=expression_data.squeeze(0))
            f.create_dataset('shape', data=shape_data.squeeze(0))
            f.create_dataset('exp', data=exp_data.squeeze(0))
            f.create_dataset('jaw', data=jaw_data.squeeze(0))
    return True


# 处理并保存
error_files = []
processed_files_count = 0

for identifier in tqdm(filtered_identifiers, desc="Processing files"):
    success = process_and_save(identifier, output_dir, error_files)
    if success:
        processed_files_count += 1

# 输出处理结果
print(f"Successfully processed files: {processed_files_count}")
print(f"Files with errors: {len(error_files)}")
if error_files:
    print("Files with errors details:")
    for identifier, error in error_files:
        print(f"Identifier: {identifier}, Error: {error}")
print("Processing complete.")
