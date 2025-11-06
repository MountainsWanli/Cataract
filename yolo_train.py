from ultralytics import YOLO

# 加载模型
model = YOLO('/home/itaer2/zxy/ultralytics-main-2/yolo11n.pt')  # 加载预训练模型

# 使用自定义配置文件训练
results = model.train(
    data='/home/itaer2/zxy/ultralytics-main-2/ultralytics/cfg/datasets/cataract.yaml',  # 配置文件路径（如果不在同目录需写全路径）
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,  # 0表示使用GPU，-1表示使用CPU
)
    