import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, get_scheduler
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import warnings
import sys
from datetime import datetime
warnings.filterwarnings("ignore")

# ================ 评估指标配置 ================
sys.path.append("/home/itaer2/zxy/shixi/code/main/blip/pycocoevalcap-master")
from bleu.bleu import Bleu
from rouge.rouge import Rouge
from meteor.meteor import Meteor
from cider.cider import Cider

# ================ 1. 配置参数 ================
config = {
    "train_csv": "/home/itaer2/zxy/cataract/datasets/split/split3/train.csv",
    "val_csv": "/home/itaer2/zxy/cataract/datasets/split/split3/val.csv",
    "test_csv": "/home/itaer2/zxy/cataract/datasets/split/split3/test.csv",
    "images_root": "/home/itaer2/zxy/cataract/datasets/images/frames3",

    "output_dir": "/home/itaer2/zxy/cataract/code/output/train3",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 30,
    "batch_size": 2,
    "accumulation_steps": 8,
    "learning_rate": 3e-5,
    "weight_decay": 1e-5,                                    
    "warmup_ratio": 0.1,
    "max_caption_length": 50,
    "use_fp16": True,
    "num_workers": 4,

    "num_beams": 3,
    "early_stopping": True,

    "image_extension": ".jpg",
    "skip_invalid_image": True
}
os.makedirs(config["output_dir"], exist_ok=True)
print(f"📌 训练配置: 设备={config['device']}, 总有效批次={config['batch_size']*config['accumulation_steps']}")

# ================ 2. 构建图片路径映射 ================
def build_image_path_map(root_dir):
    image_map = {}
    print(f"🔍 正在扫描图片根目录及子文件夹: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(config["image_extension"]):
                image_id = os.path.splitext(filename)[0]
                if image_id not in image_map:
                    image_map[image_id] = os.path.join(dirpath, filename)
                    if len(image_map) <= 5:
                        print(f"   映射示例: image_id={image_id} → 路径={os.path.join(dirpath, filename)}")
    print(f"✅ 图片路径映射构建完成，共发现 {len(image_map)} 张图片")
    return image_map

image_path_map = build_image_path_map(config["images_root"])

# ================ 3. 数据校验 ================
def validate_data():
    print("\n🔍 开始数据校验...")
    valid = True
    for name, path in [("训练集", config["train_csv"]),
                      ("验证集", config["val_csv"]),
                      ("测试集", config["test_csv"])]:
        if not os.path.exists(path):
            print(f"❌ {name}标注文件不存在: {path}")
            valid = False
        else:
            df = pd.read_csv(path)
            print(f"✅ {name}标注文件正常（{len(df)}行）: {path}")
            required_cols = ["image_id", "frame", "caption"]
            if not all(col in df.columns for col in required_cols):
                print(f"❌ {name}标注文件缺少必要列，需包含: {required_cols}")
                valid = False
    if len(image_path_map) == 0:
        print(f"❌ 未在 {config['images_root']} 中找到任何{config['image_extension']}图片")
        valid = False
    if valid and os.path.exists(config["train_csv"]):
        train_df = pd.read_csv(config["train_csv"])
        sample_ids = train_df["image_id"].head(5).tolist()
        for image_id in sample_ids:
            if image_id in image_path_map:
                print(f"✅ image_id={image_id} 找到对应图片: {image_path_map[image_id]}")
            else:
                print(f"❌ image_id={image_id} 未找到对应图片")
                valid = False
    if not valid:
        print("\n❌ 数据校验失败")
        sys.exit(1)
    print("✅ 数据校验通过！")

validate_data()

# ================ 4. 数据集类 ================
class CataractDataset(Dataset):
    def __init__(self, csv_path, image_map, processor, config):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.image_map = image_map
        self.processor = processor
        self.config = config
        self.clean_data()
        self.check_validity()
        print(f"✅ 加载 {os.path.basename(csv_path)}: {len(self.df)} 条有效数据")

    def clean_data(self):
        self.df = self.df[["image_id", "frame", "caption"]].dropna().reset_index(drop=True)
        original_count = len(self.df)
        self.df = self.df[self.df["image_id"].isin(self.image_map.keys())].reset_index(drop=True)
        print(f"   过滤无效image_id: 原始{original_count}条 → 保留{len(self.df)}条")

    def check_validity(self):
        if len(self.df) == 0:
            raise ValueError(f"数据集 {os.path.basename(self.csv_path)} 无有效数据！")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            image_id = str(row["image_id"])
            caption = str(row["caption"]).strip()
            image_path = self.image_map[image_id]
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            encoding = self.processor(
                images=image,
                text=caption,
                padding="max_length",
                truncation=True,
                max_length=self.config["max_caption_length"],
                return_tensors="pt"
            )
            data = {k: v.squeeze() for k, v in encoding.items()}
            data["image_id"] = image_id
            return data
        except Exception as e:
            if self.config.get("skip_invalid_image", True):
                return self.__getitem__((idx + 1) % len(self))
            else:
                raise

# ================ 5. 模型与数据加载 ================
processor = BlipProcessor.from_pretrained(
    "/home/itaer2/zxy/shixi/code/Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "/home/itaer2/zxy/shixi/code/Salesforce/blip-image-captioning-base"
).to(config["device"])

train_dataset = CataractDataset(config["train_csv"], image_path_map, processor, config)
val_dataset = CataractDataset(config["val_csv"], image_path_map, processor, config)
test_dataset = CataractDataset(config["test_csv"], image_path_map, processor, config) if os.path.exists(config["test_csv"]) else None

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True) if test_dataset else None

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
num_training_steps = config["epochs"] * len(train_loader) // config["accumulation_steps"]
num_warmup_steps = int(num_training_steps * config["warmup_ratio"])
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# ================ 6. 指标计算 ================
class MetricsCalculator:
    def __init__(self):
        self.bleu_scorer = Bleu(4)
        self.rouge_scorer = Rouge()
        self.meteor_scorer = Meteor()
        self.cider_scorer = Cider()
    def compute(self, predictions, references):
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        bleu_scores, _ = self.bleu_scorer.compute_score(gts, res)
        rouge_score, _ = self.rouge_scorer.compute_score(gts, res)
        meteor_score, _ = self.meteor_scorer.compute_score(gts, res)
        cider_score, _ = self.cider_scorer.compute_score(gts, res)
        return {"BLEU-1": bleu_scores[0], "BLEU-2": bleu_scores[1],
                "BLEU-3": bleu_scores[2], "BLEU-4": bleu_scores[3],
                "ROUGE-L": rouge_score, "METEOR": meteor_score, "CIDEr": cider_score}
metrics_calculator = MetricsCalculator()

# ================ 7. 训练和评估函数 ================
def _move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if k == "image_id":
            moved[k] = v
            continue
        if isinstance(v, list):
            moved[k] = v
            continue
        try:
            moved[k] = v.to(device)
        except:
            moved[k] = v
    return moved

def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:2d} [Train]")):
        batch = _move_batch_to_device(batch, config["device"])
        if config["use_fp16"]:
            with torch.cuda.amp.autocast():
                outputs = model(**{k:v for k,v in batch.items() if k!='image_id'}, labels=batch["input_ids"])
                loss = outputs.loss / config["accumulation_steps"]
            scaler.scale(loss).backward()
        else:
            outputs = model(**{k:v for k,v in batch.items() if k!='image_id'}, labels=batch["input_ids"])
            loss = outputs.loss / config["accumulation_steps"]
            loss.backward()
        if (step + 1) % config["accumulation_steps"] == 0:
            if config["use_fp16"]:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * config["accumulation_steps"]
    return total_loss / len(train_loader)

def evaluate_model(dataloader, split_name, save_results_path=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_refs, all_image_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[{split_name}] 评估中"):
            batch = _move_batch_to_device(batch, config["device"])
            outputs = model(**{k:v for k,v in batch.items() if k!='image_id'}, labels=batch["input_ids"])
            total_loss += outputs.loss.item()
            generated_ids = model.generate(
                pixel_values=batch["pixel_values"],
                max_length=config["max_caption_length"],
                num_beams=config["num_beams"],
                early_stopping=config["early_stopping"]
            )
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            refs = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)
            ids = batch.get("image_id", [])
            if not isinstance(ids, (list, tuple)):
                ids = [ids]
            n = min(len(preds), len(refs), len(ids))
            preds, refs, ids = preds[:n], refs[:n], ids[:n]
            all_preds.extend(preds)
            all_refs.extend(refs)
            all_image_ids.extend(ids)
    metrics = metrics_calculator.compute(all_preds, all_refs) if len(all_preds) > 0 else {}
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float("nan")
    print(f"\n{split_name}集生成示例:")
    for i in range(min(3, len(all_preds))):
        print(f"image_id: {all_image_ids[i]}")
        print(f"真实: {all_refs[i]}")
        print(f"预测: {all_preds[i]}")
        print("-"*80)
    if save_results_path is not None:
        import csv
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["image_id", "reference", "prediction"])
            for iid, ref, pred in zip(all_image_ids, all_refs, all_preds):
                writer.writerow([iid, ref, pred])
        print(f"✅ 已保存 {split_name} 逐样本预测到: {save_results_path}")
    return avg_loss, metrics, all_preds, all_refs, all_image_ids

# ================ 8. 训练主循环 ================
# ===================== 8. 训练主循环（仅保存BLEU-4最优模型） =====================
scaler = torch.cuda.amp.GradScaler() if config["use_fp16"] else None
best_bleu4 = -1.0
best_model_dir = os.path.join(config["output_dir"], "best_model")
log_file = os.path.join(config["output_dir"], "training_log.csv")

with open(log_file, "w") as f:
    f.write("epoch,train_loss,val_loss,BLEU-1,BLEU-2,BLEU-3,BLEU-4,ROUGE-L,METEOR,CIDEr\n")

for epoch in range(1, config["epochs"] + 1):
    train_loss = train_one_epoch(epoch)
    val_loss, val_metrics, _, _, _ = evaluate_model(val_loader, "验证")

    print(f"\nEpoch {epoch}/{config['epochs']}")
    print(f"训练损失: {train_loss:.4f}")
    print(f"验证损失: {val_loss:.4f}")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    # 写入日志
    with open(log_file, "a") as f:
        f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}")
        for k in ["BLEU-1","BLEU-2","BLEU-3","BLEU-4","ROUGE-L","METEOR","CIDEr"]:
            f.write(f",{val_metrics.get(k, float('nan')):.4f}")
        f.write("\n")

    # ✅ 仅保存BLEU-4最优模型
    if val_metrics.get("BLEU-4", -1.0) > best_bleu4:
        best_bleu4 = val_metrics["BLEU-4"]
        os.makedirs(best_model_dir, exist_ok=True)
        model.save_pretrained(best_model_dir)
        processor.save_pretrained(best_model_dir)
        print(f"🌟 最佳模型更新 (Epoch {epoch}, BLEU-4: {best_bleu4:.4f})")

print(f"\n训练完成! 最佳模型已保存至: {best_model_dir}")

# ================ 9. 测试集评估 ================
if test_loader:
    print("\n===== 测试集评估 =====")
    best_model = BlipForConditionalGeneration.from_pretrained(best_model_dir).to(config["device"])
    model = best_model
    test_results_path = os.path.join(config["output_dir"], "test_results.csv")
    test_loss, test_metrics, _, _, _ = evaluate_model(test_loader, "测试", save_results_path=test_results_path)
    print("\n测试集最终指标:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

print(f"\n训练完成! 模型保存于: {config['output_dir']}")
