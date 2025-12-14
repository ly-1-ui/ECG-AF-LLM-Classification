import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..task1.ecg_dataset import ECG_dataset
from ..task1.encoder_dummy import build_encoder

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


'''
Jsonl format:
{
    "file_name": name,
    "instruction": "请判断这个ECG信号是否有房颤？",
    "answer": "有房颤。" if label == 1 else "无房颤。",
}

Features are deprecated in the dataset, and the residual encoder code is dummy.
'''

def one_hot_to_af_label(one_hot: torch.Tensor) -> int:
    # one_hot: [N, O, A, ~] -> AF = 1 if A else 0
    return int(one_hot[2].item() == 1)


def build_split(
    dataset: ECG_dataset,
    encoder: torch.nn.Module,
    device: torch.device,
    output_path: Path,
) -> None:
    encoder.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    with torch.no_grad(), open(output_path, "w", encoding="utf-8") as f:
        for ecg, one_hot, file_name in loader:
            ecg = ecg.float().unsqueeze(1).to(device)  # [B, 1, T]
            feats = encoder(ecg).cpu().tolist()        # [B, 256]

            for feat, oh, name in zip(feats, one_hot, file_name):
                label = one_hot_to_af_label(oh)
                item = {
                    "file_name": name,
                    # "ecg_feat": feat,
                    "instruction": "请判断这个ECG信号是否有房颤？",
                    "answer": "有房颤。" if label == 1 else "无房颤。",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data"),
    )
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "dummy_llm"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder = build_encoder(device=device)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    train_dataset = ECG_dataset(str(data_root), is_train=True, cv=args.cv)
    val_dataset = ECG_dataset(str(data_root), is_train=False, cv=args.cv)

    build_split(
        train_dataset,
        encoder,
        device,
        output_dir / f"mm_instructions_train_cv{args.cv}.jsonl",
    )
    build_split(
        val_dataset,
        encoder,
        device,
        output_dir / f"mm_instructions_val_cv{args.cv}.jsonl",
    )


if __name__ == "__main__":
    main()
