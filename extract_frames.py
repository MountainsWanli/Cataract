import cv2
import pandas as pd
import os
import numpy as np
from typing import List, Tuple, Dict

# ==== 配置 ====
video_folder = "/home/itaer2/zxy/cataract/datasets/videos"          # 视频文件夹
caption_folder = "/home/itaer2/zxy/cataract/datasets/Annotations/videos2"  # 对应阶段 CSV 文件夹
output_folder = "/home/itaer2/zxy/cataract/datasets/Annotations/frames3"        # 输出每个视频的帧 CSV
frames_output_folder = "/home/itaer2/zxy/cataract/datasets/frames/frames3"  # 保存抽出的帧图片
os.makedirs(output_folder, exist_ok=True)
os.makedirs(frames_output_folder, exist_ok=True)

min_frames_per_stage = 20  # 每个阶段最少抽取的帧数
max_frames_per_stage = 40  # 每个阶段最多抽取的帧数
k = 4                      # 动态分配帧数的系数
base_key_ratio = 0.4       # 基础关键帧占比

# 场景类型对应的参数调整（动态阈值和时间间隔）
scene_params = {
    'high_dynamic': {'key_ratio': 0.5, 'diff_threshold': 0.3, 'time_interval': 1.0},
    'medium_dynamic': {'key_ratio': 0.4, 'diff_threshold': 0.2, 'time_interval': 2.0},
    'low_dynamic': {'key_ratio': 0.3, 'diff_threshold': 0.1, 'time_interval': 3.0}
}

# ==== 辅助函数 ====
def calculate_dynamic_frame_count(duration: float) -> int:
    """根据阶段时长动态计算应采样的帧数"""
    frame_count = int(k * np.sqrt(duration))
    return max(min_frames_per_stage, min(frame_count, max_frames_per_stage))

def classify_scene_dynamics(cap: cv2.VideoCapture, start_frame: int, end_frame: int) -> Tuple[str, Dict]:
    """
    分类场景动态特性（高/中/低动态）
    返回场景类型和对应的参数
    """
    # 采样少量帧计算平均差异，判断场景动态性
    total_frames = end_frame - start_frame + 1
    sample_points = min(20, total_frames)  # 最多采样20帧
    step = max(1, total_frames // sample_points)
    sample_frames = list(range(start_frame, end_frame + 1, step))
    
    diffs = []
    prev_gray = None
    
    for f in sample_frames:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, f):
            continue
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.norm(gray, prev_gray, cv2.NORM_L1) / gray.size
            diffs.append(diff)
        
        prev_gray = gray
    
    if not diffs:
        return 'low_dynamic', scene_params['low_dynamic']
    
    avg_diff = np.mean(diffs)
    
    # 根据平均差异分类场景
    if avg_diff > 0.25:  # 高动态阈值
        return 'high_dynamic', scene_params['high_dynamic']
    elif avg_diff > 0.1:  # 中动态阈值
        return 'medium_dynamic', scene_params['medium_dynamic']
    else:  # 低动态
        return 'low_dynamic', scene_params['low_dynamic']

def detect_key_frames(cap: cv2.VideoCapture, start_frame: int, end_frame: int, 
                     num_key_frames: int, diff_threshold: float) -> List[int]:
    """检测指定范围内的关键帧，使用场景自适应阈值"""
    if end_frame - start_frame < 2 or num_key_frames <= 0:
        return []
    
    total_frames = end_frame - start_frame + 1
    # 根据场景动态性调整采样密度
    step = max(1, total_frames // (num_key_frames * 4))  # 增加采样密度
    sample_frames = list(range(start_frame, end_frame + 1, step))
    
    diffs = []
    prev_gray = None
    
    for f in sample_frames:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, f):
            continue
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.norm(gray, prev_gray, cv2.NORM_L1) / gray.size
            # 只保留超过阈值的显著差异帧
            if diff > diff_threshold:
                diffs.append((f, diff))
        
        prev_gray = gray
    
    # 确保即使差异小也能选出足够的关键帧
    if len(diffs) < num_key_frames:
        # 当差异帧不足时，放宽条件选最显著的帧
        diffs.sort(key=lambda x: x[1], reverse=True)
        selected = diffs[:num_key_frames]
    else:
        # 差异帧充足时，筛选最显著的
        diffs.sort(key=lambda x: x[1], reverse=True)
        selected = diffs[:num_key_frames]
    
    # 确保关键帧在时间上分布均匀
    # 提取帧索引，解决元组问题
    selected_frames = [f for f, _ in selected]
    selected_frames.sort()
    
    if len(selected_frames) > 1:
        # 检查是否有过于密集的帧，适当分散
        spaced_frames = [selected_frames[0]]
        min_interval = total_frames // (num_key_frames * 2)  # 最小间隔
        
        for f in selected_frames[1:]:
            # 这里使用帧索引进行比较，修复了类型错误
            if f - spaced_frames[-1] > min_interval:
                spaced_frames.append(f)
        
        # 如果过滤后数量不足，补充中间帧
        if len(spaced_frames) < num_key_frames:
            # 只取帧索引进行操作
            remaining_frames = [f for f in selected_frames if f not in spaced_frames]
            spaced_frames += remaining_frames[:num_key_frames - len(spaced_frames)]
    
    return sorted(spaced_frames)

def adaptive_uniform_sampling(start_frame: int, end_frame: int, num_frames: int, 
                             key_frames: List[int], time_interval: float, fps: float) -> List[int]:
    """
    自适应均匀采样，考虑关键帧位置和时间间隔
    避免与关键帧重叠，同时保证时间分布均匀
    """
    if num_frames <= 0:
        return []
    
    # 计算基于时间间隔的理论采样点
    total_duration = (end_frame - start_frame) / fps
    num_time_based = max(1, int(total_duration / time_interval))
    
    # 生成候选采样点
    candidates = []
    if num_time_based > 0:
        time_points = np.linspace(start_frame / fps, end_frame / fps, num_time_based, endpoint=True)
        candidates = [int(t * fps) for t in time_points]
    
    # 如果候选点不足，使用帧均匀分布
    if len(candidates) < num_frames:
        candidates = np.linspace(start_frame, end_frame, num_frames * 2, dtype=int).tolist()
    
    # 过滤掉与关键帧太近的点（避免冗余）
    key_set = set(key_frames)
    filtered = []
    for f in candidates:
        # 检查是否与任何关键帧距离过近
        too_close = any(abs(f - kf) < (end_frame - start_frame) / (num_frames + len(key_frames) * 2) 
                       for kf in key_set)
        if not too_close and f not in key_set:
            filtered.append(f)
    
    # 确保数量足够
    if len(filtered) < num_frames:
        # 补充缺失的帧
        all_frames = set(range(start_frame, end_frame + 1)) - key_set
        missing = num_frames - len(filtered)
        filtered += list(all_frames - set(filtered))[:missing]
    
    # 最终选择并排序
    selected = sorted(filtered[:num_frames])
    return selected

# ==== 批量处理函数 ====
def process_video(video_path: str, caption_path: str) -> None:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing {video_name}...")

    # 读取阶段 CSV
    try:
        stages = pd.read_csv(caption_path)
        required_columns = {'start_sec', 'end_sec', 'caption'}
        if not required_columns.issubset(stages.columns):
            missing = required_columns - set(stages.columns)
            print(f"CSV文件缺少必要的列: {missing}，跳过视频 {video_name}")
            return
    except Exception as e:
        print(f"读取CSV文件出错: {str(e)}，跳过视频 {video_name}")
        return

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频属性
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"============视频的帧率为:{video_fps}=======")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / video_fps if video_fps > 0 else 0
    
    frame_records = []
    image_counter = 1

    for idx, row in stages.iterrows():
        try:
            start_sec = row["start_sec"]
            end_sec = row["end_sec"]
            caption = row["caption"]

            # 处理缺失的结束时间
            if pd.isna(end_sec) or end_sec <= start_sec:
                end_sec = min(start_sec + 5, total_duration)
            
            # 计算帧范围
            start_frame = max(0, int(start_sec * video_fps))
            end_frame = min(total_frames - 1, int(end_sec * video_fps))
            
            if end_frame <= start_frame:
                print(f"阶段 {idx+1} 帧范围无效，跳过")
                continue
                
            # 计算阶段时长和动态帧数
            duration = end_sec - start_sec
            frames_per_stage = calculate_dynamic_frame_count(duration)
            
            # 1. 分类场景动态性并获取对应参数
            scene_type, params = classify_scene_dynamics(cap, start_frame, end_frame)
            num_key_frames = max(1, int(frames_per_stage * params['key_ratio']))
            num_uniform_frames = max(1, frames_per_stage - num_key_frames)
            
            # 确保总数不超过实际可用帧数
            total_available = end_frame - start_frame + 1
            if frames_per_stage > total_available:
                frames_per_stage = total_available
                num_key_frames = max(1, int(frames_per_stage * params['key_ratio']))
                num_uniform_frames = frames_per_stage - num_key_frames

            # 2. 检测关键帧（使用场景自适应阈值）
            key_frames = detect_key_frames(
                cap, start_frame, end_frame, 
                num_key_frames, 
                params['diff_threshold']  # 动态阈值
            )
            
            # 3. 自适应均匀采样（考虑关键帧位置和时间间隔）
            uniform_frames = adaptive_uniform_sampling(
                start_frame, end_frame, 
                num_uniform_frames, 
                key_frames,
                params['time_interval'],  # 动态时间间隔
                video_fps
            )
            
            # 4. 合并、去重并排序
            all_frames = list(set(key_frames + uniform_frames))
            all_frames.sort()
            
            # 5. 智能补充缺失帧（优先选择关键帧和均匀帧之间的中间点）
            if len(all_frames) < frames_per_stage:
                missing = frames_per_stage - len(all_frames)
                # 找出帧之间的大间隙
                gaps = []
                for i in range(1, len(all_frames)):
                    gap = all_frames[i] - all_frames[i-1]
                    if gap > 1:  # 只有间隙大于1帧才考虑
                        gaps.append((all_frames[i-1], all_frames[i], gap))
                
                # 优先从最大的间隙中补充帧
                gaps.sort(key=lambda x: x[2], reverse=True)
                add_frames = []
                
                for start_gap, end_gap, _ in gaps:
                    if len(add_frames) >= missing:
                        break
                    # 在间隙中间均匀插入帧
                    insert = np.linspace(start_gap, end_gap, min(5, missing - len(add_frames) + 2), dtype=int)[1:-1]
                    add_frames.extend(insert)
                
                # 如果仍不足，从整个范围补充
                if len(add_frames) < missing:
                    remaining = missing - len(add_frames)
                    all_possible = set(range(start_frame, end_frame + 1)) - set(all_frames)
                    add_frames.extend(list(all_possible)[:remaining])
                
                all_frames += add_frames
                all_frames = sorted(list(set(all_frames)))[:frames_per_stage]

            # 处理并保存帧
            for f in all_frames:
                image_id = f"{video_name}_{image_counter:02d}"
                frame_records.append([image_id, f, caption])
                image_counter += 1

                # 保存帧图片
                if cap.set(cv2.CAP_PROP_POS_FRAMES, f):
                    ret, frame_img = cap.read()
                    if ret:
                        cv2.imwrite(os.path.join(frames_output_folder, f"{image_id}.jpg"), frame_img)
        
        except Exception as e:
            print(f"处理阶段 {idx+1} 时出错: {str(e)}，继续处理下一阶段")
            continue

    cap.release()

    # 保存 CSV
    if frame_records:
        output_csv = os.path.join(output_folder, f"{video_name}_frames.csv")
        df_frames = pd.DataFrame(
            frame_records, 
            columns=["image_id", "frame", "caption"]
        )
        df_frames.to_csv(output_csv, index=False)
        print(f"完成 {video_name}, 共提取 {len(frame_records)} 帧，保存 CSV 到 {output_csv}")
    else:
        print(f"视频 {video_name} 未提取到任何帧")


# ==== 批量处理文件夹中的视频 ====
import os

for file_name in os.listdir(video_folder):
    # 将多个后缀放在元组中传递给endswith()
    if file_name.endswith((".mp4", ".mpg")):  # 可根据需要修改视频格式
        video_path = os.path.join(video_folder, file_name)
        # 处理不同后缀的替换
        if file_name.endswith(".mp4"):
            caption_path = os.path.join(caption_folder, file_name.replace(".mp4", ".csv"))
        else:  # .mpg格式
            caption_path = os.path.join(caption_folder, file_name.replace(".mpg", ".csv"))

        if os.path.exists(caption_path):
            process_video(video_path, caption_path)
        else:
            print(f"未找到对应的 caption CSV: {caption_path}，跳过视频 {file_name}")
print("全部视频处理完成！")
    