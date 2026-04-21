# DecTalk3D：分层解耦引导的情感可控 VQ-VAE 3D 说话人脸生成方法<br>DecTalk3D: Emotion-controllable 3D Talking Face Generation with Hierarchical Disentanglement-guided VQ-VAE

## 简介

本仓库为论文 **《分层解耦引导的情感可控 VQ-VAE 3D 说话人脸生成方法》** 的代码实现。

DecTalk3D 以 VQ-VAE 为基础，将人脸运动特征划分为顶层与底层两个层次，并引入 **语音、文本、身份向量** 作为条件信息，对人脸特征进行分层解耦与条件引导建模，从而提升 3D 说话人脸生成的重建精度、情感表达稳定性和可控性。

- 论文页面：http://cjig.cn/zh/article/doi/10.11834/jig.250451/
- 论文 DOI：https://doi.org/10.11834/jig.250451
- 仓库地址：https://github.com/chen114514sheng/DecTalk3D

## 方法概述

本文方法包含两个阶段：

- **重建阶段**：分层解耦重建。顶层特征在身份向量与文本条件引导下建模，底层特征在顶层特征和外部条件共同约束下建模。
- **生成阶段**：在重建阶段离散表示的基础上，结合语音、文本和身份条件生成人脸运动参数。

### 重建阶段

![stage1](images/图1.png)

### 生成阶段

![stage2](images/图2.png)

## 项目结构

```text
DecTalk3D/
├── DataProcess/           # 数据预处理与数据集划分
├── FLAME/                 # FLAME 相关文件与模板
├── VQVAE2/                # 第一阶段：分层解耦 VQ-VAE
├── Generation/            # 第二阶段：条件生成模型
├── Render0.py             # 生成说话人脸视频
├── Render1.py             # 与其他模型结果对比
├── Quality.py             # 质量分析与 heatmap
├── Experiments/           # 条件交换实验（可选）
├── AuxClassifier/         # 辅助分类器（可选）
└── config.yaml            # 路径与训练/预测配置
```

## 运行环境

当前项目环境：

- Python 3.8.18
- 系统 CUDA 12.4（`nvcc 12.4.99`）
- 当前环境未通过 conda 安装 `cudatoolkit`，依赖系统 CUDA

建议使用 Linux + NVIDIA GPU 环境运行。

## 数据集

本项目使用：

- **MEAD**：语音与表情数据  
  官网：https://wywu.github.io/projects/MEAD/MEAD.html
- **3DMEAD**：由 MEAD 相关数据进一步处理得到的 3D 人脸运动数据  
  下载位置：https://github.com/radekd91/inferno/tree/release/EMOTE/inferno_apps/TalkingHead/data_processing
- **TA-MEAD**：文本描述数据

### 数据预处理

```bash
python DataProcess/mead0.py
python DataProcess/mead1.py
```

## 配置说明

项目中的数据路径、FLAME 路径、模型权重路径和训练参数统一通过 `config.yaml` 设置。

至少建议检查以下内容：

- `train_file_path`
- `val_file_path`
- `test_file_path`
- `flame_model`
- `static_landmark_embedding`
- `dynamic_landmark_embedding`
- `predict.vqvae_dir`
- `predict.generation_dir`
- `predict.save_path`
- `stage1.checkpoint_dir`
- `stage2.checkpoint_dir`

## 训练

### 第一阶段：训练分层解耦 VQ-VAE

```bash
python VQVAE2/Train.py
```

### 第二阶段：训练生成模型

```bash
python Generation/Train.py
```

## 测试与生成

### 第一阶段：评估重建效果

```bash
python VQVAE2/Predict.py
```

### 第二阶段：评估生成效果并保存结果

```bash
python Generation/Predict.py
```

### 渲染视频与对比结果

```bash
python Render0.py
python Render1.py
python Quality.py
```

对比结果：https://www.bilibili.com/video/BV1ivdQB1EZW

## 条件交换实验（可选）

若需要交换实验中的定量评估，请先训练辅助分类器：

```bash
python AuxClassifier/train_emotion.py
python AuxClassifier/train_identity.py
```

再使用脚本：

```bash
python Experiments/build_swap_pairs.py
python Experiments/run_stage1_swap.py --pair_type text_emotion --deduplicate_reverse
python Experiments/run_stage2_swap.py --pair_type text_emotion --deduplicate_reverse
python Experiments/eval_swap_metrics.py
python Experiments/render_swap_vis.py --stage all --pair_type all
python Experiments/render_swap_video.py --stage all --pair_type all
```

重建阶段：https://www.bilibili.com/video/BV1ZvdQB1EMV

生成阶段：https://www.bilibili.com/video/BV1qvdQB1EjL

## 引用

如果本项目对你的研究有帮助，请引用本文：

```bibtex
@article{chen2026dectalk3d,
  title   = {分层解耦引导的情感可控VQ-VAE 3D说话人脸生成方法},
  author  = {陈胜 and 孙强 and 朱霞天},
  journal = {中国图象图形学报},
  year    = {2026},
  pages   = {1--15},
  doi     = {10.11834/jig.250451},
  url     = {http://cjig.cn/zh/article/doi/10.11834/jig.250451/}
}
```
