import os
import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO

# ========== 配置 ==========
video_path = "/home/itaer2/zxy/cataract/external/datasets/testvideo/video4.mpg"
yolo_weights = "/home/itaer2/zxy/ultralytics-main-2/runs/detect/cataract/weights/best.pt"
stage_csv = "/home/itaer2/zxy/cataract/external/datasets/Annotations/videos/videos2/video4.csv"
output_csv = "./auto_sam2_box_prompts_external.csv"
preview_dir = "./preview_frames_external"  # 帧预览保存路径
os.makedirs(preview_dir, exist_ok=True)

fps = 25  # 视频帧率
conf_thres = 0.4  # YOLO检测阈值

# ========== 1. 加载模型与阶段表 ==========
print("🚀 正在加载YOLO模型...")
model = YOLO(yolo_weights)
df = pd.read_csv(stage_csv)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"❌ 无法打开视频: {video_path}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 视频分辨率: {w}x{h}")

# ========== 2. 自动生成 Box Prompts ==========
records = []  # 保存所有结果

for idx, row in df.iterrows():
    start_sec, end_sec, caption = row["start_sec"], row["end_sec"], row["caption"]
    clip_idx = idx
    print(f"\n🩺 处理阶段 {clip_idx}: {caption}")

    # 三帧采样：起始 / 中间 / 结束
    frame_times = np.linspace(start_sec, end_sec, num=3)

    for sec in frame_times:
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ 无法读取 {sec:.2f}s 的帧，跳过。")
            continue

        # YOLO 检测
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

        # 过滤低置信度
        mask = confs > conf_thres
        boxes = boxes[mask]
        cls_ids = cls_ids[mask]

        if len(boxes) == 0:
            print(f"❌ 阶段 [{caption}] 帧 {frame_idx} 无检测结果。")
            continue

        # 绘制检测框用于预览
        preview_frame = frame.copy()
        for i, (box, cls_id) in enumerate(zip(boxes, cls_ids)):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview_frame, f"cls{int(cls_id)}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 保存 CSV 记录
            records.append({
                "clip_idx": clip_idx,
                "frame_idx": frame_idx,
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "label": 1,  # SAM2 通常 1 表示正样本
                "caption": caption
            })

        # 保存预览图
        preview_path = os.path.join(preview_dir, f"clip{clip_idx}_frame{frame_idx}.jpg")
        cv2.imwrite(preview_path, preview_frame)
        print(f"🖼️ 已保存检测预览帧: {preview_path}")

cap.release()

# ========== 3. 保存为 CSV ==========
out_df = pd.DataFrame(records)
out_df.to_csv(output_csv, index=False)
print(f"\n💾 已保存 CSV: {output_csv}")
print(f"📊 共生成 {len(out_df)} 条检测提示框。")
print(f"🖼️ 预览图文件夹: {preview_dir}")

# ========== 4. 示例：SAM2加载方式 ==========
# df = pd.read_csv(output_csv)
# for _, row in df.iterrows():
#     predictor.add_new_points_or_box(
#         frame_idx=int(row["frame_idx"]),
#         obj_id=2,
#         box=np.array([row[["x1","y1","x2","y2"]].values], dtype=np.float32)
#     )
