import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.task2.llm_model import MultimodalLLM


DEVICE = "cuda"
CNN_CKPT = "./model.pth"
DATA_PATH = "llm_dataset.pt"

BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-4


def collate_fn(batch):
    ecg = torch.stack([b["ecg_feature"] for b in batch])
    ecg = ecg.unsqueeze(1).float()
    instr = [b["instruction"] for b in batch]
    ans = [b["answer"] for b in batch]
    return ecg, instr, ans


def main():
    data = torch.load(DATA_PATH)
    loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = MultimodalLLM(
        cnn_ckpt=CNN_CKPT,
        device=DEVICE,
        use_lora=True
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for ecg, instr, ans in pbar:
            ecg = ecg.to(DEVICE)

            out = model(ecg, instr, ans)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    model.llm.save_pretrained("./lora_output")
    print("✅ LoRA 训练完成")


if __name__ == "__main__":
    main()
