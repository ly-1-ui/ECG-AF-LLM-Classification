import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader

from .llm_model import ECGQwenForAF, DEFAULT_QWEN_NAME, apply_lora_to_llm
from .train import _ecg_preprocess

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class EvalDataset(Dataset):
    def __init__(self, jsonl_path: str, mat_dir: str, ecg_len: int, downsample: int):
        self.items: List[Dict[str, Any]] = []
        self.mat_dir = Path(mat_dir)
        self.ecg_len = ecg_len
        self.downsample = downsample
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                instr = obj["instruction"]
                prompt = instr + "\n答："
                self.items.append(
                    {
                        "file_name": obj["file_name"],
                        "prompt": prompt,
                        "gt_answer": obj["answer"],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def _load_mat(self, file_name: str) -> np.ndarray:
        mat_path = self.mat_dir / f"{file_name}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"ECG mat not found: {mat_path}")
        data = io.loadmat(str(mat_path))["val"]
        sig = np.asarray(data).reshape(-1)
        return sig

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        sig = self._load_mat(it["file_name"])
        sig = _ecg_preprocess(
            sig,
            out_len=self.ecg_len,
            downsample=self.downsample,
            random_crop=False,
        )
        ecg = torch.from_numpy(sig).unsqueeze(0)
        return {"ecg": ecg, "prompt": it["prompt"], "gt_answer": it["gt_answer"]}

def collate_fn(batch, tokenizer, max_length: int = 512):
    ecg = torch.stack([b["ecg"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    return (
        ecg,
        enc["input_ids"],
        enc["attention_mask"],
        [b["gt_answer"] for b in batch],
        prompts,
    )

def load_checkpoint(model: ECGQwenForAF, ckpt_path: str) -> ECGQwenForAF:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "trainable_state_dict" in state:
        model.load_state_dict(state["trainable_state_dict"], strict=False)
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        raise KeyError("Checkpoint missing 'trainable_state_dict' or 'model_state_dict'.")

    return model

def extract_label_from_text(text: str) -> int:
    negative_phrases = [
        "无房颤", "没房颤", "没有房颤", "不是房颤", "无心房颤动", "没有心房颤动",
        "no af", "not af"
    ]
    positive_phrases = [
        "有房颤", "是房颤", "心房颤动", "af"
    ]

    text = text.strip()
    t = text.lower()

    for p in negative_phrases:
        if p in text or p in t:
            return 0
    for p in positive_phrases:
        if p in text or p in t:
            return 1

    return 2

def extract_label_from_gt(gt: str) -> int:
    gt = gt.strip()
    if "有房颤" in gt:
        return 1
    if "无房颤" in gt:
        return 0
    return 0

def evaluate(
    ckpt_path: str,
    val_path: str,
    mat_dir: str,
    llm_name: str = DEFAULT_QWEN_NAME,
    encoder_ckpt: Optional[str] = None,
    ecg_len: int = 2400,
    downsample: int = 3,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGQwenForAF(
        llm_name=llm_name,
        ecg_ch_in=1,
        ecg_len=ecg_len,
        ecg_encoder_ckpt=encoder_ckpt,
        freeze_encoder=True,
    )
    model = apply_lora_to_llm(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer

    dataset = EvalDataset(
        val_path,
        mat_dir=mat_dir,
        ecg_len=ecg_len,
        downsample=downsample,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    preds: List[int] = []
    gts: List[int] = []

    for idx, (ecg, input_ids, attn_mask, gt_answers, prompts) in enumerate(loader):
        with torch.no_grad():
            ecg = ecg.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            inputs_embeds, full_attention_mask = model.prepare_inputs_for_generation(
                ecg=ecg,
                input_ids=input_ids,
                attention_mask=attn_mask,
            )
            full_attention_mask = full_attention_mask.to(device)

            generated_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=4,
                do_sample=False,
                num_beams=1,
            )

        gen_text = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        pred_label = extract_label_from_text(gen_text)  # 0/1/2
        gt_label = extract_label_from_gt(gt_answers[0]) # 0/1

        preds.append(pred_label)
        gts.append(gt_label)

        if idx % 20 == 0:
            print(f"[{idx}/{len(dataset)}]")
            print("Prompt:", prompts[0])
            print("Generated:", gen_text)
            print("GT answer:", gt_answers[0])
            print("-" * 60, flush=True)

    preds_tensor = torch.tensor(preds)
    gts_tensor = torch.tensor(gts)

    acc = (preds_tensor == gts_tensor).float().mean().item()

    tp = ((preds_tensor == 1) & (gts_tensor == 1)).sum().item()
    fp = ((preds_tensor == 1) & (gts_tensor == 0)).sum().item()
    fn = ((preds_tensor == 0) & (gts_tensor == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("Accuracy:", acc) 
    print("Precision:", precision) # TP / (TP + FP)
    print("Recall:", recall)       # TP / (TP + FN)
    print("F1:", f1)               # 2 * (Precision * Recall) / (Precision + Recall)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--mat-dir", type=str, required=True)
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--llm-name", type=str, default=DEFAULT_QWEN_NAME)
    parser.add_argument("--ecg-len", type=int, default=2400)
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        ckpt_path=args.ckpt,
        val_path=args.val,
        mat_dir=args.mat_dir,
        llm_name=args.llm_name,
        encoder_ckpt=args.encoder_ckpt,
        ecg_len=args.ecg_len,
        downsample=args.downsample,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
