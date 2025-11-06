import pandas as pd
import os
import random

# ==== 配置 ====
frames_csv_folder = "/home/itaer2/zxy/cataract/datasets/Annotations/frames3"   # 每个视频的 frames CSV 文件夹
output_folder = "/home/itaer2/zxy/cataract/datasets/split/split3"       # 保存 train/val/test CSV
os.makedirs(output_folder, exist_ok=True)

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
assert train_ratio + val_ratio + test_ratio == 1.0

# ==== 获取所有视频 CSV 文件 ====
all_csv_files = [f for f in os.listdir(frames_csv_folder) if f.endswith(".csv")]
all_csv_files.sort()
random.shuffle(all_csv_files)

num_videos = len(all_csv_files)
train_end = int(num_videos * train_ratio)
val_end = int(num_videos * (train_ratio + val_ratio))

train_files = all_csv_files[:train_end]
val_files = all_csv_files[train_end:val_end]
test_files = all_csv_files[val_end:]

# ==== 合并生成 train/val/test CSV ====
def merge_csv(file_list):
    dfs = []
    for f in file_list:
        df = pd.read_csv(os.path.join(frames_csv_folder, f))
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["image_id", "frame", "caption"])

train_df = merge_csv(train_files)
val_df = merge_csv(val_files)
test_df = merge_csv(test_files)

# ==== 保存 CSV ====
train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_folder, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

print(f"划分完成，train: {len(train_df)}行, val: {len(val_df)}行, test: {len(test_df)}行")
