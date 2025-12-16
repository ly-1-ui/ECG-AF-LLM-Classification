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
            
        # DEBUG
        # if not printed:
        #     printed = True
        #     # decode input_ids
        #     print("=== DEBUG: decoded input_ids ===")
        #     print(tokenizer.decode(input_ids[0]))

        #     # decode labels (answer)
        #     label_ids = labels[0]
        #     answer_token_ids = label_ids[label_ids != -100]
        #     print("=== DEBUG: decoded labels (answer only) ===")
        #     print(tokenizer.decode(answer_token_ids))

        #     # also print raw answer text for comparison
        #     print("=== DEBUG: raw answer ===")
        #     print(texts[0])

        #     print("=" * 60, flush=True)

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
    num_epochs: int = 3,
    lr: float = 1e-4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    resume: Optional[str] = None,
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    num_update_steps_per_epoch = max(len(train_loader), 1)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )

    start_epoch = 0
    global_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        if "trainable_state_dict" in ckpt:
            model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            raise KeyError("Checkpoint missing 'trainable_state_dict' or 'model_state_dict'.")

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        did_step = 0
        zero_sup_batches = 0
        ran_debug = False

        for step, batch in enumerate(train_loader):
            ecg, input_ids, attention_mask, labels = batch
            ecg = ecg.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            sup_mask = (labels != -100)  # [B, T_text]
            sup_tokens = int(sup_mask.sum().item())
            if sup_tokens == 0:
                zero_sup_batches += 1
            
            # DEBUG
            # if epoch == start_epoch and step == 0:
            #     for i in range(min(4, labels.size(0))):
            #         ans_ids = labels[i][labels[i] != -100]
            #         print(f"[DEBUG] gt supervised text[{i}]:", tokenizer.decode(ans_ids))
            #         print(f"[DEBUG] gt supervised ids[{i}]:", ans_ids.tolist())
                
            # DEBUG
            # if epoch == start_epoch and step == 0:
                # per_sample = sup_mask.sum(dim=1).tolist()
                # print(f"[DEBUG] supervised tokens per sample (first batch): {per_sample}", flush=True)
                # print(
                #     f"[DEBUG] supervised tokens total={sup_tokens} | "
                #     f"min={min(per_sample)} max={max(per_sample)} avg={sum(per_sample)/max(len(per_sample),1):.2f}",
                #     flush=True,
                # )

            # DEBUG
            # if (not ran_debug) and sup_tokens > 0:
            #     ran_debug = True
            #     model.eval()
            #     with torch.no_grad():
            #         out_real = model(
            #             ecg=ecg,
            #             input_ids=input_ids,
            #             attention_mask=attention_mask,
            #             labels=labels,
            #         )
            #         out_zero = model(
            #             ecg=torch.zeros_like(ecg),
            #             input_ids=input_ids,
            #             attention_mask=attention_mask,
            #             labels=labels,
            #         )

            #         diff = (out_real.logits - out_zero.logits).abs()  # [B, T_total, V]
            #         sup_text = (labels != -100)  # [B, T_text]

            #         pad = torch.zeros((sup_text.size(0), 1), dtype=torch.bool, device=sup_text.device)
            #         sup_total = torch.cat([pad, sup_text], dim=1)

            #         sup_shift = sup_total[:, 1:]     # [B, T_total-1]
            #         diff_shift = diff[:, :-1, :]     # [B, T_total-1, V]

            #         diff_per_pos = diff_shift.mean(dim=-1)  # [B, T_total-1]
            #         diff_sup = diff_per_pos[sup_shift].mean().item() if sup_shift.any() else float("nan")

            #         idxs = sup_shift[0].nonzero(as_tuple=True)[0]
            #         diff_last = diff_per_pos[0, int(idxs[-1].item())].item() if idxs.numel() > 0 else float("nan")

            #         print(
            #             f"[DEBUG epoch {epoch+1}] loss_real={float(out_real.loss):.6f} "
            #             f"loss_zero={float(out_zero.loss):.6f} "
            #             f"diff_sup={diff_sup:.6e} diff_last={diff_last:.6e}",
            #             flush=True,
            #         )
            #     model.train()

            outputs = model(
                ecg=ecg,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            
            # DEBUG
            # if (step == 0) and (epoch == start_epoch):
            #     with torch.no_grad():
            #         logits = outputs.logits  # [B, T_total, V]
            #         # pad one False for ECG then shift
            #         sup_text = (labels != -100)
            #         pad = torch.zeros((sup_text.size(0), 1), dtype=torch.bool, device=sup_text.device)
            #         sup_total = torch.cat([pad, sup_text], dim=1)
            #         sup_shift = sup_total[:, 1:]
            #         pred_ids = logits[:, :-1, :].argmax(dim=-1)  # [B, T_total-1]

            #         # show first sample's predicted supervised tokens
            #         idxs = sup_shift[0].nonzero(as_tuple=True)[0]
            #         p = pred_ids[0, idxs]
            #         print("[DEBUG] pred supervised tokens:", model.tokenizer.decode(p), flush=True)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            did_step += 1
            total_loss += float(loss.item())

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # DEBUG
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
                # check 1
                zero_outputs = model(
                    ecg=torch.zeros_like(ecg), 
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )

                val_loss += outputs.loss.item()
                zero_loss += zero_outputs.loss.item() # DEBUG

        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_zero_loss = zero_loss / max(len(val_loader), 1)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
            f" | zero_loss{avg_zero_loss:.4f}"
        )
        model.train()

        if (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs:
            time_str = datetime.now().strftime("%Y%m%d-%H%M")
            ckpt_path = output_dir / f"checkpoint-epoch{epoch + 1}-{time_str}.pt"
            torch.save(
                {
                    "trainable_state_dict": get_trainable_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "meta": {
                        "llm_name": llm_name,
                        "ecg_len": ecg_len,
                        "downsample": downsample,
                        "lora_r": lora_r,
                        "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout,
                    },
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ecg-len", type=int, default=2400)
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--llm-name", type=str, default=DEFAULT_QWEN_NAME)
    parser.add_argument("--resume", type=str, default=None)
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
        num_epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume=args.resume,
    )
