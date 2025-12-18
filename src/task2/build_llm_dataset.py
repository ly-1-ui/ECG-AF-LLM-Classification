import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.task1.ecg_dataset import ECG_dataset          # 你已有的 Dataset
from src.task1.encoder import Mscnn, load_mscnn_checkpoint, freeze_module


# =========================
# 1. 配置参数
# =========================
BASE_DIR = "./data"          # ECG 数据根目录
CNN_CKPT = "./model.pth"    # Task1 训练好的 CNN 权重
SAVE_PATH = "llm_dataset.pt"

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTION_TEXT = "请判断这个ECG信号是否有房颤？"


# =========================
# 2. 标签 → 文本答案
# =========================
def label_to_answer(one_hot: torch.Tensor) -> str:
    """
    Task1 标签是 4 类，这里只关心 A (房颤)
    A 对应 one-hot: [0,0,1,0]
    """
    if one_hot[2] == 1:
        return "有房颤。"
    else:
        return "无房颤。"


# =========================
# 3. 构建 CNN 编码器
# =========================
def build_cnn_encoder():
    model = Mscnn(
        ch_in=1,
        ch_out=1,
        use_stream2=True
    )
    load_mscnn_checkpoint(model, CNN_CKPT, map_location=DEVICE)
    freeze_module(model)          # ❗ 冻结 CNN
    model.eval()
    model.to(DEVICE)
    return model


# =========================
# 4. 构建 Task2.1 数据集
# =========================
def build_instruction_dataset():
    dataset = ECG_dataset(
        base_file=BASE_DIR,
        cv=0,
        is_train=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    encoder = build_cnn_encoder()

    instruction_data = []

    with torch.no_grad():
        for ecg, one_hot, file_name in tqdm(loader):
            ecg = ecg.unsqueeze(1).float().to(DEVICE)  # [B, 1, T]

            # 1️⃣ CNN 特征
            features = encoder.forward_features(ecg)  # [B, feature_dim]

            for i in range(features.size(0)):
                sample = {
                    "ecg_feature": features[i].cpu(),   # Tensor
                    "instruction": INSTRUCTION_TEXT,
                    "answer": label_to_answer(one_hot[i]),
                    "file_name": file_name[i]
                }
                instruction_data.append(sample)

    return instruction_data


# =========================
# 5. 保存数据
# =========================
if __name__ == "__main__":
    data = build_instruction_dataset()

    torch.save(data, SAVE_PATH)

    print(f"✅ Task 2.1 数据集已生成，共 {len(data)} 条样本")
    print(f"📦 保存路径: {SAVE_PATH}")
