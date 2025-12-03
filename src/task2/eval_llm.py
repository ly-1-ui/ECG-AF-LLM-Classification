import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from .llm_model import ECGQwenForAF

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class EvalDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                instr = obj["instruction"]
                prompt = instr + "\n答："
                self.items.append(
                    {
                        "ecg_feat": obj["ecg_feat"],
                        "prompt": prompt,
                        "gt_answer": obj["answer"],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

def collate_fn(batch, tokenizer, max_length: int = 256):
    ecg_feats = torch.tensor(
        [b["ecg_feat"] for b in batch], dtype=torch.float32
    )
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
        ecg_feats,
        enc["input_ids"],
        enc["attention_mask"],
        [b["gt_answer"] for b in batch],
        prompts,
    )

def load_checkpoint(model: ECGQwenForAF, ckpt_path: str) -> ECGQwenForAF:
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    return model

def extract_label_from_text(text: str) -> int:
    text = text.strip()
    if "有房颤" in text:
        return 1
    if "无房颤" in text:
        return 0
    lower = text.lower()
    if "af" in lower:
        return 1
    if "no af" in lower:
        return 0
    return 0

def extract_label_from_gt(gt: str) -> int:
    gt = gt.strip()
    if "有房颤" in gt:
        return 1
    if "无房颤" in gt:
        return 0
    return 0

def build_inputs_embeds(
    model: ECGQwenForAF,
    ecg_feat: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
):
    ecg_feat = ecg_feat.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    llm_dtype = next(model.llm.parameters()).dtype

    ecg_feat = ecg_feat.to(model.proj.weight.dtype)
    ecg_emb = model.proj(ecg_feat)
    ecg_emb = ecg_emb.to(llm_dtype).unsqueeze(1)

    text_emb = model.llm.get_input_embeddings()(input_ids)
    if text_emb.dtype != llm_dtype:
        text_emb = text_emb.to(llm_dtype)

    inputs_embeds = torch.cat([ecg_emb, text_emb], dim=1)

    ecg_mask = torch.ones(
        ecg_emb.size(0),
        1,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    full_attention_mask = torch.cat([ecg_mask, attention_mask], dim=1)

    return inputs_embeds, full_attention_mask

def evaluate(
    ckpt_path: str,
    val_path: str,
    ecg_dim: int = 256,
    max_length: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGQwenForAF(ecg_dim=ecg_dim)
    model = load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer

    dataset = EvalDataset(val_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    preds: List[int] = []
    gts: List[int] = []

    for ecg_feat, input_ids, attn_mask, gt_answers, prompts in loader:
        ecg_feat = ecg_feat.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            inputs_embeds, full_attention_mask = build_inputs_embeds(
                model, ecg_feat, input_ids, attn_mask, device
            )

            generated_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=16,
                do_sample=False,
                num_beams=1,
            )

        gen_text = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        pred_label = extract_label_from_text(gen_text)
        gt_label = extract_label_from_gt(gt_answers[0])

        preds.append(pred_label)
        gts.append(gt_label)

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
    parser.add_argument("--ecg-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        ckpt_path=args.ckpt,
        val_path=args.val,
        ecg_dim=args.ecg_dim,
        max_length=args.max_length,
    )
