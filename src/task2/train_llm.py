import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

from .llm_model import ECGQwenForAF, DEFAULT_QWEN_NAME

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ECGAFLLMDataset(Dataset):
    """Dataset for LLM fine-tuning with ECG features and instruction-answer pairs."""

    def __init__(self, jsonl_path: str):
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # text: instruction + answer as one sequence
                text = f"{obj['instruction']}\n答：{obj['answer']}"
                self.items.append(
                    {
                        "ecg_feat": obj["ecg_feat"],
                        "text": text,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def make_collate_fn(tokenizer, max_length: int = 256):
    def collate_fn(batch: List[Dict[str, Any]]):
        ecg_feats = torch.tensor(
            [b["ecg_feat"] for b in batch], dtype=torch.float32
        )  # [B, ecg_dim]
        texts = [b["text"] for b in batch]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.clone()

        return ecg_feats, input_ids, attention_mask, labels

    return collate_fn


def apply_lora_to_llm(
    model: ECGQwenForAF,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> ECGQwenForAF:
    # Freeze base LLM weights
    for p in model.llm.parameters():
        p.requires_grad = False

    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],  # Qwen attention projections
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()
    return model


def train(
    train_path: str,
    val_path: str,
    output_dir: str,
    ecg_dim: int = 256,
    llm_name: str = DEFAULT_QWEN_NAME,
    batch_size: int = 4,
    num_epochs: int = 3,
    lr: float = 1e-4,
    max_length: int = 256,
    grad_accum_steps: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGQwenForAF(ecg_dim=ecg_dim, llm_name=llm_name)
    model = apply_lora_to_llm(model)
    model.to(device)

    tokenizer = model.tokenizer

    train_dataset = ECGAFLLMDataset(train_path)
    val_dataset = ECGAFLLMDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )

    # Only train LoRA parameters and projection layer
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    num_update_steps_per_epoch = max(len(train_loader) // grad_accum_steps, 1)
    max_train_steps = num_epochs * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            ecg_feat, input_ids, attention_mask, labels = batch
            ecg_feat = ecg_feat.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                ecg_feat=ecg_feat,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum_steps

        avg_train_loss = total_loss / len(train_loader)

        # Simple evaluation on val set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ecg_feat, input_ids, attention_mask, labels = batch
                ecg_feat = ecg_feat.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(
                    ecg_feat=ecg_feat,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
        )

        model.train()

        time_str = datetime.now().strftime("%Y%m%d-%H%M")
        ckpt_path = output_dir / f"checkpoint-epoch{epoch + 1}-{time_str}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "dummy_llm"),
    )
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "outputs" / "llm_cv0"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_root = Path(args.data_root)
    train_path = data_root / f"mm_instructions_train_cv{args.cv}.jsonl"
    val_path = data_root / f"mm_instructions_val_cv{args.cv}.jsonl"

    train(
        train_path=str(train_path),
        val_path=str(val_path),
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        grad_accum_steps=args.grad_accum,
    )
