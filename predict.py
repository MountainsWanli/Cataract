import cv2
import torch
import csv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ========= 配置部分 =========
video_path = "/home/itaer2/zxy/cataract/external/datasets/videos/video1.mp4"  # 输入视频路径
model_path = "/home/itaer2/zxy/cataract/code/output/train2/best_model"  # 训练好的BLIP权重
output_csv = "/home/itaer2/zxy/cataract/external/datasets/test/video1.csv"  # 输出的CSV文件
fps_sampling = 2.0   # 每秒抽多少帧 (例如 1fps 表示每秒取1帧)
# ===========================

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)

# 打开视频
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
sample_interval = int(video_fps / fps_sampling)

raw_captions = []
frame_idx = 0

print("开始处理视频...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % sample_interval == 0:
        # 当前时间（秒）
        time_sec = frame_idx / video_fps

        # 转换为 PIL Image 供 BLIP 使用
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 模型推理
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(** inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True)

        raw_captions.append((time_sec, caption))
        print(f"{int(time_sec)}s {caption}")

    frame_idx += 1

cap.release()

# ========== 合并相邻相同描述 ==========
merged = []
if raw_captions:
    start_time = raw_captions[0][0]
    prev_caption = raw_captions[0][1]

    for i in range(1, len(raw_captions)):
        cur_time, cur_caption = raw_captions[i]
        if cur_caption != prev_caption:
            # 当前描述和之前不同 -> 结束上一个时间段
            merged.append({
                'start_sec': round(start_time, 2),
                'end_sec': round(cur_time, 2),
                'caption': prev_caption
            })
            start_time = cur_time
            prev_caption = cur_caption

    # 最后一段
    end_time = raw_captions[-1][0] + (1 / fps_sampling)  # 根据采样频率计算结束时间
    merged.append({
        'start_sec': round(start_time, 2),
        'end_sec': round(end_time, 2),
        'caption': prev_caption
    })

# ========== 保存为CSV文件 ==========
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['start_sec', 'end_sec', 'caption'])
    writer.writeheader()  # 写入表头
    for item in merged:
        writer.writerow(item)

print(f"\n处理完成！结果已保存到: {output_csv}")
