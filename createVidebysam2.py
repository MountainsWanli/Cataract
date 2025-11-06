import os
import cv2
import numpy as np
import torch
import glob
import shutil
import pandas as pd
from sam2.build_sam import build_sam2_video_predictor

# ===================== 1. 配置参数 =====================
clips_dir = "/home/itaer2/zxy/cataract/code/sam2-main/video_clip_external"
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
output_root = "./outputs_clips_25fps_external"
os.makedirs(output_root, exist_ok=True)

# YOLO 检测结果 CSV 文件
yolo_csv_path = "/home/itaer2/zxy/cataract/code/sam2-main/sam2/auto_sam2_box_prompts_external.csv"
fps = 25
clip_length_sec = 60
clip_length_frames = int(fps * clip_length_sec)

# 眼部提示点（中心点坐标）- 根据原始框计算
# 原始框 [206, 2, 1095, 672] 的中心点: (x_center, y_center)
eye_center = np.array([[(206 + 1095) / 2, (2 + 672) / 2]], dtype=np.float32)  # 形状为 (1, 2)
eye_label = np.array([1], dtype=np.int64)  # 1 表示前景

# ===================== 2. 从 CSV 读取检测框生成 Prompt =====================
df_boxes = pd.read_csv(yolo_csv_path)
print(f"📑 已加载 {len(df_boxes)} 条 YOLO 检测记录")

# 计算每个检测框属于哪个 clip
df_boxes["clip_index"] = (df_boxes["frame_idx"] // clip_length_frames).astype(int)
df_boxes["clip_local_frame"] = (df_boxes["frame_idx"] % clip_length_frames).astype(int)

# 分组构建 prompt 字典
instrument_boxes_by_clip = {}
for _, row in df_boxes.iterrows():
    clip_idx = int(row["clip_index"])
    frame_idx = int(row["clip_local_frame"])
    box = np.array([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32)
    if clip_idx not in instrument_boxes_by_clip:
        instrument_boxes_by_clip[clip_idx] = []
    instrument_boxes_by_clip[clip_idx].append({
        "frame_idx": frame_idx,
        "box": box,
        "caption": row.get("caption", "")
    })

print(f"✅ 已生成 {len(instrument_boxes_by_clip)} 个 clip 的提示框。")

# ===================== 3. 设备配置 =====================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ===================== 4. 工具函数 =====================
def adjust_box(box, w, h):
    """调整框坐标至视频分辨率范围内"""
    adjusted = box.copy()
    adjusted[0] = max(0, min(adjusted[0], w - 1))
    adjusted[1] = max(0, min(adjusted[1], h - 1))
    adjusted[2] = max(adjusted[0] + 1, min(adjusted[2], w))
    adjusted[3] = max(adjusted[1] + 1, min(adjusted[3], h))
    return adjusted

def adjust_points(points, w, h):
    """调整点坐标至视频分辨率范围内"""
    adjusted = points.copy()
    adjusted[:, 0] = np.clip(adjusted[:, 0], 0, w - 1)  # x坐标限制
    adjusted[:, 1] = np.clip(adjusted[:, 1], 0, h - 1)  # y坐标限制
    return adjusted

# ===================== 5. 主处理函数 =====================
def process_single_clip(clip_path, clip_index, predictor):
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]
    output_dir = os.path.join(output_root, clip_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n===== 开始处理片段 {clip_index}: {clip_name} =====")

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {clip_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 分辨率 {w}x{h}, 帧数 {frame_count}, FPS={fps:.1f}")
    cap.release()

    # 1️⃣ 提取视频帧（保留BGR格式）
    temp_dir = os.path.join(output_dir, "frames")
    os.makedirs(temp_dir, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(os.path.join(temp_dir, f"{i}.jpg"), frame)  # 不转换RGB
    cap.release()

    # 2️⃣ 初始化 SAM2 状态
    inference_state = predictor.init_state(video_path=temp_dir)
    predictor.reset_state(inference_state)

    # 3️⃣ 添加眼部提示点（替换原来的提示框）
    eye_center_adj = adjust_points(eye_center, w, h)  # 调整点坐标至图像范围内
    _, eye_obj_ids, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=min(5, frame_count-1),  # 在前5帧中添加提示
        obj_id=1,
        points=eye_center_adj,  # 使用中心点作为提示
        labels=eye_label  # 标记为前景
    )
    print(f"✅ 添加眼部提示点 (Obj 1)")

    # 4️⃣ 添加 YOLO 检测框作为器械提示
    instrument_prompts = []
    if clip_index in instrument_boxes_by_clip:
        for prompt in instrument_boxes_by_clip[clip_index]:
            box_adj = adjust_box(prompt["box"], w, h)
            _, obj_ids, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt["frame_idx"],
                obj_id=2,
                box=box_adj
            )
            instrument_prompts.append(prompt)
            print(f"✅ 添加器械提示框 (Obj 2) 帧 {prompt['frame_idx']}")
    else:
        print(f"⚠️ 当前片段 {clip_index} 无检测框提示")

    # 5️⃣ 可视化提示框与提示点
    print("🎨 可视化提示框中...")
    vis_frame_indices = {min(5, frame_count-1)} | {p["frame_idx"] for p in instrument_prompts}
    cap = cv2.VideoCapture(clip_path)
    for frame_idx in vis_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ 可视化帧 {frame_idx} 读取失败，跳过")
            continue
        
        # 绘制眼部中心点（绿色）
        center_x, center_y = int(eye_center_adj[0][0]), int(eye_center_adj[0][1])
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)  # 实心圆
        cv2.putText(frame, "Eye (Obj1)",
                    (center_x + 15, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制器械框（红色）
        for prompt in instrument_prompts:
            if prompt["frame_idx"] == frame_idx:
                box = adjust_box(prompt["box"], w, h)
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)
                cv2.putText(frame, "Instrument (Obj2)",
                            (int(box[0]), int(box[1])-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        vis_save_path = os.path.join(output_dir, f"prompt_vis_frame_{frame_idx}.jpg")
        cv2.imwrite(vis_save_path, frame)
    cap.release()
    print(f"✅ 提示点可视化完成，保存至 {output_dir}")

    # 6️⃣ 掩码传播
    print("🔄 开始掩码传播...")
    video_segments = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        masks = {}
        for i, obj_id in enumerate(obj_ids):
            mask = (mask_logits[i] > 0.5).squeeze(0).cpu().numpy()
            if mask.any():
                masks[obj_id] = mask
        if masks:
            video_segments[frame_idx] = masks

    # 7️⃣ 输出分割视频
    out_path = os.path.join(output_dir, f"{clip_name}_segmented.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    colors = {1: (0, 255, 0), 2: (0, 0, 255)}
    for i in range(frame_count):
        frame = cv2.imread(os.path.join(temp_dir, f"{i}.jpg"))
        if i in video_segments:
            mask_layer = np.zeros_like(frame, dtype=np.uint8)
            for obj_id, mask in video_segments[i].items():
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                mask_layer[mask] = colors.get(obj_id, (255, 255, 255))
            frame = cv2.addWeighted(frame, 0.6, mask_layer, 0.4, 0)
        writer.write(frame)
    writer.release()
    print(f"🎥 已保存分割视频: {out_path}")

    shutil.rmtree(temp_dir)
    print(f"🗑️ 清理临时帧目录 {temp_dir}")

# ===================== 6. 批量处理 =====================
if __name__ == "__main__":
    print("🔧 加载SAM2模型中...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    print("✅ SAM2加载完成")

    clip_paths = sorted(glob.glob(os.path.join(clips_dir, "video4_clip_*.mp4")),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
    if not clip_paths:
        print(f"❌ 未找到任何视频片段: {clips_dir}")
        exit()

    for idx, clip_path in enumerate(clip_paths):
        process_single_clip(clip_path, idx, predictor)

    print(f"\n🎉 全部 {len(clip_paths)} 个片段处理完成！输出目录: {output_root}")