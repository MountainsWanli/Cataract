Cataract 项目说明文档
**
项目简介
本项目包含两大核心模块（blip 和 Yolo11Sam2），聚焦于白内障相关的计算机视觉任务，涵盖数据处理、模型训练、目标检测与分割、视频生成等功能，适用于白内障相关的图像 / 视频分析场景。
项目结构
Cataract/
├── blip/                  # BLIP 相关模块（数据处理+caption任务）
│   ├── processdata/       # 数据预处理工具
│   │   ├── extract_frames.py  # 视频帧提取脚本
│   │   └── split.py        # 数据集划分脚本（训练/验证/测试集）
│   ├── merge_caption.py    # caption 结果合并脚本
│   ├── predict.py          # BLIP 模型预测脚本
│   └── train.py            # BLIP 模型训练脚本
└── Yolo11Sam2/            # Yolo11+Sam2 目标检测与分割模块
    ├── create_prompt.py    # Sam2 提示词创建脚本
    ├── createVidebysam2.py # Sam2 视频生成脚本
    └── yolo_train.py       # Yolo11 模型训练脚本

核心功能
1. blip 模块
数据预处理：支持从视频中提取帧（extract_frames.py）、划分数据集（split.py）
模型训练：基于 BLIP 训练图像描述（caption）模型（train.py）
预测与结果合并：生成图像描述并合并结果（predict.py、merge_caption.py）
2. Yolo11Sam2 模块
目标检测：使用 Yolo11 训练白内障相关目标检测模型（yolo_train.py）
实例分割：通过 Sam2 实现精准分割，支持创建自定义提示词（create_prompt.py）
视频生成：基于分割结果生成处理后的视频（createVidebysam2.py）
环境依赖
建议使用 Python 3.8+，核心依赖如下（需手动安装）：
# 基础依赖
pip install torch torchvision opencv-python numpy pandas

# BLIP 相关依赖
pip install transformers datasets pillow

# Yolo11 依赖
pip install ultralytics

# Sam2 依赖（参考官方安装指南）
pip install segment-anything-2

使用流程
快速上手步骤
克隆项目到本地：
git clone <项目仓库地址>
cd Cataract

数据准备：
放入原始视频 / 图像数据到指定目录（建议在项目根目录创建 data/ 文件夹）
运行 blip/processdata/extract_frames.py 提取视频帧
运行 blip/processdata/split.py 划分数据集
模型训练：
BLIP 模型：修改 blip/train.py 中的配置（数据路径、超参数等），运行训练脚本
Yolo11 模型：修改 Yolo11Sam2/yolo_train.py 中的配置，运行训练脚本
预测 / 推理：
运行 blip/predict.py 生成图像描述，使用 merge_caption.py 合并结果
运行 Yolo11Sam2/create_prompt.py 创建提示词，结合 createVidebysam2.py 生成分割视频
注意事项
运行脚本前需根据实际需求修改配置（如数据路径、模型参数、输出目录等）
Sam2 模型使用需提前下载预训练权重（参考 Sam2 官方文档）
建议为不同模块创建独立虚拟环境，避免依赖冲突
若遇到数据读取错误，请检查文件路径是否正确配置
联系方式
若有问题或建议，欢迎联系项目维护者！
