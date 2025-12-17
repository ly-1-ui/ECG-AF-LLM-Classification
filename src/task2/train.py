import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from .llm_model import ECGQwenForAF, DEFAULT_QWEN_NAME, apply_lora_to_llm

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _crop_or_pad_1d(x: np.ndarray, length: int, random_crop: bool) -> np.ndarray:
    n = x.shape[0]
    if n < length:
        pad = length - n
        return np.pad(x, (0, pad), mode="constant")
    if n > length:
        if random_crop:
            start = np.random.randint(0, n - length + 1)
        else:
            start = (n - length) // 2
        return x[start : start + length]
    return x


def _ecg_preprocess(
    sig_1d: np.ndarray,
    out_len: int = 3000,
    downsample: int = 3,
    random_crop: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    x = sig_1d.astype(np.float32)
    if downsample > 1:
        x = x[::downsample]
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + eps)
    x = _crop_or_pad_1d(x, out_len, random_crop=random_crop)
    return x


class ECGAFLLMDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        mat_dir: str,
        ecg_len: int = 2400,
        downsample: int = 3,
        is_train: bool = True,
    ):
        self.items: List[Dict[str, Any]] = []
        self.mat_dir = Path(mat_dir)
        self.ecg_len = ecg_len
        self.downsample = downsample
        self.is_train = is_train

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(
                    {
                        "file_name": obj["file_name"],
                        "instruction": obj["instruction"],
                        "answer": obj["answer"],
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
            random_crop=self.is_train,
        )
        ecg = torch.from_numpy(sig).unsqueeze(0)
        prompt = f"{it['instruction']}\n答："
        full_text = f"{prompt}{it['answer']}"
        return {"ecg": ecg, "prompt": prompt, "text": full_text}


def make_collate_fn(tokenizer, max_length: int = 512):
    printed = False  # closure flag
    
    def collate_fn(batch: List[Dict[str, Any]]):
        nonlocal printed
        
        ecg = torch.stack([b["ecg"] for b in batch], dim=0)
        prompts = [b["prompt"] for b in batch]
        texts = [b["text"] for b in batch]

        enc_full = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        enc_prompt = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        prompt_attn = enc_prompt["attention_mask"]
        prompt_lens = prompt_attn.sum(dim=1)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        for i, plen in enumerate(prompt_lens.tolist()):
            plen = int(plen)
            labels[i, :plen] = -100

        return ecg, input_ids, attention_mask, labels

    return collate_fn


def get_trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    full_sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in full_sd.items() if k in trainable_names}


def train(
    train_path: str,
    val_path: str,
    mat_dir: str,
    output_dir: str,
    llm_name: str = DEFAULT_QWEN_NAME,
    encoder_ckpt: Optional[str] = None,
    ecg_len: int = 2400,
    downsample: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    resume: Optional[str] = None,
    ecg_token_count: int = 16,
    stage1_epochs: Optional[int] = None,
    stage2_epochs: int = 0,
    stage1_lr: Optional[float] = None,
    stage2_adapter_lr: Optional[float] = None,
    stage2_lora_lr: Optional[float] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1_epochs = max(stage1_epochs, 0)
    stage2_epochs = max(stage2_epochs, 0)
    total_epochs = stage1_epochs + stage2_epochs

    if stage1_lr is None:
        stage1_lr = lr
    if stage2_adapter_lr is None:
        stage2_adapter_lr = stage1_lr * 0.1
    if stage2_lora_lr is None:
        stage2_lora_lr = stage1_lr

    model = ECGQwenForAF(
        llm_name=llm_name,
        ecg_ch_in=1,
        ecg_len=ecg_len,
        ecg_encoder_ckpt=encoder_ckpt,
        freeze_encoder=True,
        ecg_token_count=ecg_token_count,
    )
    for p in model.llm.parameters():
        p.requires_grad = False
    model.to(device)

    tokenizer = model.tokenizer

    train_dataset = ECGAFLLMDataset(
        jsonl_path=train_path,
        mat_dir=mat_dir,
        ecg_len=ecg_len,
        downsample=downsample,
        is_train=True,
    )
    val_dataset = ECGAFLLMDataset(
        jsonl_path=val_path,
        mat_dir=mat_dir,
        ecg_len=ecg_len,
        downsample=downsample,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
        pin_memory=torch.cuda.is_available(),
    )

    adapter_params = list(model.ecg_adapter.parameters()) + list(model.proj.parameters())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_epoch = 0
    global_step = 0
    stage1_start_epoch = 0
    stage2_start_epoch = 0
    resume_stage = None
    resume_opt_state = None
    resume_sched_state = None
    lora_initialized = False

    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        resume_stage = int(ckpt.get("stage", 1))
        if resume_stage == 2 and not lora_initialized:
            model = apply_lora_to_llm(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            model.to(device)
            lora_initialized = True

        if "trainable_state_dict" in ckpt:
            model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            raise KeyError("Checkpoint missing 'trainable_state_dict' or 'model_state_dict'.")

        global_epoch = int(ckpt.get("global_epoch", ckpt.get("epoch", 0)))
        global_step = int(ckpt.get("global_step", 0))

        if resume_stage == 1:
            stage1_start_epoch = int(ckpt.get("stage_epoch", ckpt.get("epoch", 0)))
            stage2_start_epoch = 0
        elif resume_stage == 2:
            stage1_start_epoch = stage1_epochs
            stage2_start_epoch = int(ckpt.get("stage_epoch", 0))
        else:
            stage1_start_epoch = int(ckpt.get("stage_epoch", 0))
            stage2_start_epoch = 0

        resume_opt_state = ckpt.get("optimizer_state_dict")
        resume_sched_state = ckpt.get("scheduler_state_dict")

    stage1_start_epoch = min(stage1_start_epoch, stage1_epochs)
    stage2_start_epoch = min(stage2_start_epoch, stage2_epochs)

    def build_scheduler(optimizer, stage_epoch_count: int):
        num_update_steps_per_epoch = max(len(train_loader), 1)
        max_train_steps = stage_epoch_count * num_update_steps_per_epoch
        if max_train_steps == 0:
            max_train_steps = 1
        num_warmup_steps = int(0.1 * max_train_steps)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

    def run_stage(stage_idx: int, start_epoch: int, max_epoch: int, optimizer, scheduler):
        nonlocal global_epoch, global_step
        if max_epoch == 0 or start_epoch >= max_epoch:
            return

        stage_name = f"stage{stage_idx}"
        model.train()

        for local_epoch in range(start_epoch, max_epoch):
            total_loss = 0.0
            did_step = 0
            zero_sup_batches = 0

            for batch in train_loader:
                ecg, input_ids, attention_mask, labels = batch
                ecg = ecg.to(device, non_blocking=True)
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                sup_mask = (labels != -100)
                if int(sup_mask.sum().item()) == 0:
                    zero_sup_batches += 1

                outputs = model(
                    ecg=ecg,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                did_step += 1
                total_loss += float(loss.item())

            avg_train_loss = total_loss / max(len(train_loader), 1)

            if did_step == 0:
                print("WARNING: optimizer.step() never called in this epoch. Check len(train_loader).", flush=True)
            if zero_sup_batches > 0:
                print(
                    f"WARNING: {zero_sup_batches}/{len(train_loader)} batches have 0 supervised tokens "
                    f"(all labels=-100). Try increasing max_length.",
                    flush=True,
                )

            model.eval()
            val_loss = 0.0
            zero_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    ecg, input_ids, attention_mask, labels = batch
                    ecg = ecg.to(device, non_blocking=True)
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(
                        ecg=ecg,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    zero_outputs = model(
                        ecg=torch.zeros_like(ecg),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    val_loss += outputs.loss.item()
                    zero_loss += zero_outputs.loss.item()

            avg_val_loss = val_loss / max(len(val_loader), 1)
            avg_zero_loss = zero_loss / max(len(val_loader), 1)
            overall_epoch = global_epoch + 1
            total_epoch_denom = max(total_epochs, 1)
            stage_epoch_idx = local_epoch + 1
            print(
                f"[{stage_name}] Epoch {overall_epoch}/{total_epoch_denom} (stage {stage_epoch_idx}/{max_epoch}) | "
                f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | zero_loss{avg_zero_loss:.4f}"
            )
            model.train()

            time_str = datetime.now().strftime("%Y%m%d-%H%M")
            ckpt_path = output_dir / (
                f"checkpoint-{stage_name}-epoch{overall_epoch}-{ecg_token_count}token-{time_str}.pt"
            )
            if stage_idx > 1:
                torch.save(
                    {
                        "stage": stage_idx,
                        "stage_epoch": stage_epoch_idx,
                        "epoch": overall_epoch,
                        "global_epoch": overall_epoch,
                        "global_step": global_step,
                        "trainable_state_dict": get_trainable_state_dict(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "meta": {
                            "llm_name": llm_name,
                            "ecg_len": ecg_len,
                            "downsample": downsample,
                            "lora_r": lora_r,
                            "lora_alpha": lora_alpha,
                            "lora_dropout": lora_dropout,
                            "stage1_epochs": stage1_epochs,
                            "stage2_epochs": stage2_epochs,
                            "stage1_lr": stage1_lr,
                            "stage2_adapter_lr": stage2_adapter_lr,
                            "stage2_lora_lr": stage2_lora_lr,
                        },
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

            global_epoch += 1

    # Stage 1: only ECG adapter/proj without LoRA
    if stage1_epochs > 0:
        stage1_optimizer = torch.optim.AdamW(adapter_params, lr=stage1_lr)
        stage1_scheduler = build_scheduler(stage1_optimizer, stage1_epochs)
        if resume_stage == 1 and resume_opt_state is not None:
            stage1_optimizer.load_state_dict(resume_opt_state)
        if resume_stage == 1 and resume_sched_state is not None:
            stage1_scheduler.load_state_dict(resume_sched_state)
        run_stage(1, stage1_start_epoch, stage1_epochs, stage1_optimizer, stage1_scheduler)

    # Stage 2: enable LoRA and train with smaller adapter LR
    if stage2_epochs > 0:
        if not lora_initialized:
            model = apply_lora_to_llm(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            model.to(device)
            lora_initialized = True

        lora_params = [p for p in model.llm.parameters() if p.requires_grad]
        param_groups = []
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": stage2_adapter_lr})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": stage2_lora_lr})

        stage2_optimizer = torch.optim.AdamW(param_groups)
        stage2_scheduler = build_scheduler(stage2_optimizer, stage2_epochs)
        if resume_stage == 2 and resume_opt_state is not None:
            stage2_optimizer.load_state_dict(resume_opt_state)
        if resume_stage == 2 and resume_sched_state is not None:
            stage2_scheduler.load_state_dict(resume_sched_state)
        run_stage(2, stage2_start_epoch, stage2_epochs, stage2_optimizer, stage2_scheduler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "llm_cv0"),
    )
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument(
        "--mat-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "training2017"),
    )
    parser.add_argument(
        "--encoder-ckpt",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "model.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "llm_cv0"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stage1-epochs", type=int, default=None,
                        help="Number of epochs for stage 1 (defaults to --epochs if unset)")
    parser.add_argument("--stage2-epochs", type=int, default=0,
                        help="Number of epochs for stage 2 training with LoRA")
    parser.add_argument("--stage1-lr", type=float, default=None,
                        help="Learning rate for stage 1 (defaults to --lr)")
    parser.add_argument("--stage2-adapter-lr", type=float, default=None,
                        help="Learning rate for ECG adapter/proj in stage 2")
    parser.add_argument("--stage2-lora-lr", type=float, default=None,
                        help="Learning rate for LoRA params in stage 2")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ecg-len", type=int, default=2400)
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--llm-name", type=str, default=DEFAULT_QWEN_NAME)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ecg_token_count", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = Path(args.data_root)
    train_path = data_root / f"mm_instructions_train_cv{args.cv}_posx10.jsonl"
    val_path = data_root / f"mm_instructions_val_cv{args.cv}.jsonl"

    train(
        train_path=str(train_path),
        val_path=str(val_path),
        mat_dir=args.mat_dir,
        output_dir=args.output_dir,
        llm_name=args.llm_name,
        encoder_ckpt=args.encoder_ckpt,
        ecg_len=args.ecg_len,
        downsample=args.downsample,
        batch_size=args.batch_size,
        lr=args.lr,
        stage1_lr=args.stage1_lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume=args.resume,
        ecg_token_count=args.ecg_token_count,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage2_adapter_lr=args.stage2_adapter_lr,
        stage2_lora_lr=args.stage2_lora_lr,
    )
