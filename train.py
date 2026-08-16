"""
训练脚本。默认在 SyntheticShapesDataset 上跑，几分钟内（CPU 也可以）就能看到
模型从瞎猜收敛到能正确描述图像里的颜色/形状/位置。

用法：
    python train.py --epochs 8 --batch_size 64

跑完后：
- checkpoints/mini_vlm.pt 保存权重
- 训练过程中会打印若干个验证样本的"真实描述 vs 模型生成"，直观看到收敛情况
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import SyntheticShapesDataset
from model import MiniVLM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_samples", type=int, default=4000)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def evaluate_samples(model, dataset, device, n=4):
    model.eval()
    idxs = torch.randperm(len(dataset))[:n]
    bos_id, eos_id = dataset.stoi["<bos>"], dataset.stoi["<eos>"]
    for i in idxs:
        image, tokens = dataset[i]
        gt = dataset.decode(tokens)
        gen_tokens, _ = model.generate(image.unsqueeze(0).to(device), bos_id, eos_id)
        pred = dataset.decode(gen_tokens[0].cpu())
        print(f"  真实: {gt:35s} | 生成: {pred}")


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"[device] {device}")

    full_ds = SyntheticShapesDataset(n_samples=args.n_samples, img_size=args.img_size, max_len=args.max_len)
    val_size = max(1, int(0.1 * len(full_ds)))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    vocab_size = len(full_ds.stoi)
    pad_id = full_ds.stoi["<pad>"]

    model = MiniVLM(
        vocab_size=vocab_size, img_size=args.img_size, patch_size=8,
        d_model=args.d_model, n_heads=4, n_layers=3, d_ff=args.d_model * 2,
        max_len=args.max_len, pad_id=pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量: {n_params / 1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, tokens in train_loader:
            images, tokens = images.to(device), tokens.to(device)
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            logits, _ = model(images, input_tokens)
            loss = criterion(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"[epoch {epoch}/{args.epochs}] train_loss={avg_loss:.4f}")
        evaluate_samples(model, val_ds.dataset, device)

    torch.save({"model_state": model.state_dict(), "stoi": full_ds.stoi, "itos": full_ds.itos,
                "args": vars(args)}, "checkpoints/mini_vlm.pt")
    print("[done] 权重已保存到 checkpoints/mini_vlm.pt")


if __name__ == "__main__":
    main()
