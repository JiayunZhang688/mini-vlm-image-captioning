"""
可视化 cross-attention：生成一条描述的同时，把模型在预测每个词时"看图像哪个 patch
看得最多"画出来，直观展示视觉-语言对齐效果。这是面试时最容易讲清楚、也最有说服力
的一张图——比只说"我训练了一个VLM"更有信服力。

用法：
    python visualize_attention.py --checkpoint checkpoints/mini_vlm.pt --n 3
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import SyntheticShapesDataset
from model import MiniVLM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/mini_vlm.pt")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", type=str, default="samples/attention.png")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    stoi, itos, cfg = ckpt["stoi"], ckpt["itos"], ckpt["args"]

    ds = SyntheticShapesDataset(n_samples=args.n, img_size=cfg["img_size"], max_len=cfg["max_len"], seed=123)
    ds.stoi, ds.itos = stoi, itos  # 用训练时的词表，保证 id 对齐

    model = MiniVLM(
        vocab_size=len(stoi), img_size=cfg["img_size"], patch_size=8,
        d_model=cfg["d_model"], n_heads=4, n_layers=3, d_ff=cfg["d_model"] * 2,
        max_len=cfg["max_len"], pad_id=stoi["<pad>"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_side = cfg["img_size"] // 8  # patch_size=8

    fig, axes = plt.subplots(1, args.n, figsize=(4 * args.n, 4.5))
    if args.n == 1:
        axes = [axes]

    for i in range(args.n):
        image, tokens = ds[i]
        gt = ds.decode(tokens)
        gen_tokens, attn_steps = model.generate(
            image.unsqueeze(0), stoi["<bos>"], stoi["<eos>"], return_cross_attn=True
        )
        pred = ds.decode(gen_tokens[0])

        # 取生成最后一步的 attention，reshape 成 (n_side, n_side) 的热力图
        if len(attn_steps) > 0:
            last_attn = attn_steps[-1][0].detach().numpy().reshape(n_side, n_side)
        else:
            last_attn = np.zeros((n_side, n_side))

        img_np = image.permute(1, 2, 0).numpy()
        ax = axes[i]
        ax.imshow(img_np)
        heat = np.kron(last_attn, np.ones((8, 8)))  # 放大回像素尺寸
        ax.imshow(heat, cmap="jet", alpha=0.45, extent=(0, cfg["img_size"], cfg["img_size"], 0))
        ax.set_title(f"GT: {gt}\nPred: {pred}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[done] 可视化结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
