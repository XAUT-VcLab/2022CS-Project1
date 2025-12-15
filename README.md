# 分层解耦引导的情感可控VQ-VAE 3D说话人脸生成方法
## 简介（Introduction）
项目位置：https://github.com/chen114514sheng/Project1

本项目结合语音、文本、身份向量，在VQ-VAE结构的基础上，通过分层解耦的方法生成说话人脸视频。
## 方法概述（Method）
项目整体流程如下：

第一阶段
<img width="18817" height="10108" alt="图1" src="https://github.com/user-attachments/assets/832d7659-f43c-4a2d-8f24-b5df1cb460ea" />
第二阶段
<img width="18942" height="7808" alt="图2" src="https://github.com/user-attachments/assets/819c6991-25b6-4731-bfa9-b0ee7e6b8bee" />
## 数据集
3D人脸数据来自3DMEAD，语音数据来自MEAD，文本数据来自TA-MEAD。

数据集通过DataProcess/mead0.py预处理，通过DataProcess/mead1.py划分。
