from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, clips_array
from PIL import Image, ImageDraw, ImageFont
import os


def create_labeled_image(label, width, height=50):
    """生成带顶部标签的黑色背景图"""
    img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 30)  # 使用更清晰的字体
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, 10), label, fill="white", font=font)
    img_path = f"temp_{label}.png"
    img.save(img_path)
    return img_path


# 模型配置列表（名称，路径，是否需要test_前缀）
models = [
    ("Ground Truth", "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/test_video", False),
    ("STtb", "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/STtb_video", False),
    ("CodeTalker", "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/CodeTalker_video", True),
    ("FaceDiffuser", "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/FaceDiffuser_video", True),
    ("FaceFormer", "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/FaceFormer_video", True),
    ("ProbTalk3D", "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/ProbTalk3D_video", True)
]
output_path = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project1/predict/_video"
os.makedirs(output_path, exist_ok=True)

# 获取基准视频列表（以Ground Truth为基准）
base_path = models[0][1]
for base_file in os.listdir(base_path):
    if not base_file.endswith(".mp4"):
        continue

    clips = []
    temp_files = []

    for label, model_path, needs_prefix in models:
        # 动态生成目标文件名
        target_file = f"test_{base_file}" if needs_prefix else base_file
        video_path = os.path.join(model_path, target_file)

        if not os.path.exists(video_path):
            print(f"⚠️ 缺失文件: {video_path}")
            continue

        # 加载视频并创建标签
        try:
            clip = VideoFileClip(video_path)
            label_img = create_labeled_image(label, clip.w)
            text_clip = ImageClip(label_img).set_duration(clip.duration)

            # 合成带标签的视频片段
            composite = CompositeVideoClip([
                clip.set_position(("center", 50)),
                text_clip.set_position(("center", 0))
            ], size=(clip.w, clip.h + 50))

            clips.append(composite)
            temp_files.extend([label_img, video_path])  # 记录临时文件

        except Exception as e:
            print(f"❌ 处理 {label} 失败: {str(e)}")
            continue

    # 检查是否收集到全部6个视频
    if len(clips) != 6:
        print(f"⏩ 跳过不完整的合成: {base_file} (找到 {len(clips)}/6 个视频)")
        continue

    # 构建两行三列布局
    grid = clips_array([
        clips[:3],  # 第一行
        clips[3:]  # 第二行
    ])

    # 输出合成视频
    output_file = os.path.join(output_path, base_file)
    grid.write_videofile(
        output_file,
        codec="libx264",
        fps=25,
        threads=8,  # 多线程加速
        preset="slow",  # 高质量编码
        audio_codec="aac"
    )

    # 在写入文件后立即关闭并清理
    grid.close()
    for clip in clips:
        clip.close()  # 显式关闭合成clip
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    # 添加强制垃圾回收
    import gc
    gc.collect()

print("✅ 合成完成！")
