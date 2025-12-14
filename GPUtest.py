import subprocess
import torch
import time
import psutil

"""
获取服务器GPU信息
"""

def get_gpu_info():
    # 使用 torch 获取 GPU 基本信息
    print("GPU 基本信息（通过 PyTorch）：")
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved_memory = torch.cuda.memory_reserved(i) / 1024 ** 3

        print(f"设备 {i} : {device_name}")
        print(f"  总内存: {total_memory:.2f} GB")
        print(f"  已用内存: {allocated_memory:.2f} GB")
        print(f"  缓存内存: {reserved_memory:.2f} GB")
        print()


def get_gpu_usage():
    # 使用 nvidia-smi 获取 GPU 使用情况
    print("GPU 使用情况（通过 nvidia-smi）：")
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv,nounits'],
        stdout=subprocess.PIPE)
    gpu_info = result.stdout.decode('utf-8').strip().split('\n')

    for info in gpu_info[1:]:
        index, name, total_mem, used_mem, free_mem = info.split(',')
        print(f"设备 {index} : {name}")
        print(f"  总内存: {total_mem} MiB")
        print(f"  已用内存: {used_mem} MiB")
        print(f"  空闲内存: {free_mem} MiB")
        print()


def get_gpu_processes():
    # 使用 nvidia-smi 获取正在使用 GPU 的进程信息
    print("正在使用 GPU 的进程信息（通过 nvidia-smi）：")
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,nounits'],
        stdout=subprocess.PIPE)
    process_info = result.stdout.decode('utf-8').strip().split('\n')

    if len(process_info) > 1:
        for info in process_info[1:]:
            gpu_uuid, pid, process_name, used_memory = info.split(',')
            print(f"GPU UUID: {gpu_uuid}")
            print(f"  进程 ID: {pid}")
            print(f"  进程名称: {process_name}")
            print(f"  使用内存: {used_memory} MiB")
            print()
    else:
        print("没有进程正在使用 GPU。")


def monitor_memory(interval=5):
    while True:
        # 获取系统内存信息
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total / (1024 ** 3)  # 转换为 GB
        used_memory = memory_info.used / (1024 ** 3)    # 转换为 GB
        available_memory = memory_info.available / (1024 ** 3)  # 转换为 GB

        # 打印系统内存信息
        print(f"Total Memory: {total_memory:.2f} GB")
        print(f"Used Memory: {used_memory:.2f} GB")
        print(f"Available Memory: {available_memory:.2f} GB")

        # 获取 GPU 显存信息（如果使用 GPU）
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为 GB
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)    # 转换为 GB
            print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
            print(f"Reserved GPU Memory: {reserved_memory:.2f} GB")

        print("-" * 50)

        # 等待指定的时间间隔
        time.sleep(interval)


def main():

    get_gpu_info()
    get_gpu_usage()
    get_gpu_processes()
    """
    # 启动内存监控
    monitor_memory(interval=5)
    """

if __name__ == "__main__":
    main()
