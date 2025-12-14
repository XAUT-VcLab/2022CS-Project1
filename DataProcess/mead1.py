import os
import shutil
from sklearn.model_selection import train_test_split

"""
数据集划分
/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/mead
"""

# 数据集目录路径
output_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/mead/mead"
output_train_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/train/train"
output_val_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/train/val"
output_test_dir = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/test"

# 创建训练集、验证集和测试集目录
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# 获取所有文件列表
all_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

# 首先按0.2的比例划分出测试集+验证集（20%），剩下的80%为训练集
train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)

# 然后在剩下的20%中，再按0.5的比例划分为验证集和测试集（各10%）
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# 复制文件到相应的目录
for f in train_files:
    shutil.copy2(os.path.join(output_dir, f), os.path.join(output_train_dir, f))

for f in val_files:
    shutil.copy2(os.path.join(output_dir, f), os.path.join(output_val_dir, f))

for f in test_files:
    shutil.copy2(os.path.join(output_dir, f), os.path.join(output_test_dir, f))

# 输出结果
print(f"Total files: {len(all_files)}")
print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Testing files: {len(test_files)}")
