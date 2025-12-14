import json
import matplotlib.pyplot as plt


def plot_loss(record_file, output_dir):
    # 读取记录文件
    with open(record_file, 'r') as f:
        records = json.load(f)

    # 提取训练和验证损失
    train_losses = records.get("train_losses", [])
    val_losses = records.get("val_losses", [])


    # 提取最小损失
    best_epoch_train = records.get("best_epoch_train", -1)
    best_loss_train = records.get("best_loss_train", float('inf'))
    best_epoch_val = records.get("best_epoch_val", -1)
    best_loss_val = records.get("best_loss_val", float('inf'))

    epochs = range(1, len(train_losses) + 1)

    # 创建绘图
    plt.figure(figsize=(10, 6))
    # 绘制训练和验证损失曲线
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')

    # 标注最佳epoch和损失值
    plt.axvline(x=best_epoch_train + 1, color='r', linestyle='--', label=f'Best Epoch Train: {best_epoch_train + 1}')
    plt.axhline(y=best_loss_train, color='g', linestyle='--', label=f'Best Loss Train: {best_loss_train:.8f}')
    plt.axvline(x=best_epoch_val + 1, color='r', linestyle='--', label=f'Best Epoch Val: {best_epoch_val + 1}')
    plt.axhline(y=best_loss_val, color='g', linestyle='--', label=f'Best Loss Val: {best_loss_val:.8f}')

    # 添加图例和标签
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Val Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # 保存绘图
    #output_file = os.path.join(output_dir, 'loss_plot.png')
    #os.makedirs(output_dir, exist_ok=True)
    #plt.savefig(output_file)
    #print(f"损失变化图已保存到 {output_file}")
    plt.show()


# python plot.py
if __name__ == '__main__':
    # VQVAE2 Generation _STtb _tb _STb _b
    record_file = '/home/chensheng/1Project/Project1/Generation/checkpoints0/loss_record.json'
    output_dir = '/home/chensheng/1Project/Project1/Generation/checkpoints0'
    plot_loss(record_file, output_dir)
