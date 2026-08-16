"""
数据集模块。

1) SyntheticShapesDataset —— 完全离线、程序生成的"图像-文字"配对数据：
   在一张纯色背景图上画一个随机颜色/形状/位置的几何图形，配上对应的英文描述模板，
   例如 "a red circle on the left"。这不需要下载任何数据，几秒钟就能生成上万条样本，
   足够验证整条 VLM pipeline（图像编码 -> 跨模态对齐 -> 文本生成）是否跑得通。

2) 真实数据集（简历里应该用这个）：项目跑通之后，把 SyntheticShapesDataset 换成
   Flickr8k / COCO Captions 的 DataLoader 即可，模型代码完全不用改，因为接口一致
   （返回 image tensor + token id 序列）。README 里有具体替换说明。
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 60, 60), "green": (60, 180, 90), "blue": (60, 100, 220),
    "yellow": (230, 200, 60), "purple": (150, 70, 200),
}
SHAPES = ["circle", "square", "triangle"]
POSITIONS = ["left", "right", "top", "bottom", "center"]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def build_vocab():
    words = set()
    for c in COLORS:
        words.add(c)
    words.update(SHAPES)
    words.update(POSITIONS)
    words.update(["a", "on", "the"])
    vocab = SPECIAL_TOKENS + sorted(words)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def _position_coords(position, img_size, shape_r):
    c = img_size // 2
    off = img_size // 4
    return {
        "left": (c - off, c),
        "right": (c + off, c),
        "top": (c, c - off),
        "bottom": (c, c + off),
        "center": (c, c),
    }[position]


def _draw_shape(draw, shape, color, cx, cy, r):
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:  # triangle
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


class SyntheticShapesDataset(Dataset):
    def __init__(self, n_samples=4000, img_size=32, max_len=8, seed=0):
        self.n_samples = n_samples
        self.img_size = img_size
        self.max_len = max_len
        self.stoi, self.itos = build_vocab()
        rng = random.Random(seed)
        # 预先采样好每条样本的属性，保证可复现
        self.records = []
        for _ in range(n_samples):
            color = rng.choice(list(COLORS.keys()))
            shape = rng.choice(SHAPES)
            position = rng.choice(POSITIONS)
            self.records.append((color, shape, position))

    def __len__(self):
        return self.n_samples

    def _render(self, color, shape, position):
        img = Image.new("RGB", (self.img_size, self.img_size), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        r = self.img_size // 6
        cx, cy = _position_coords(position, self.img_size, r)
        _draw_shape(draw, shape, COLORS[color], cx, cy, r)
        arr = np.asarray(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _tokenize(self, color, shape, position):
        words = ["a", color, shape, "on", "the", position]
        ids = [self.stoi["<bos>"]] + [self.stoi[w] for w in words] + [self.stoi["<eos>"]]
        ids = ids[: self.max_len]
        ids = ids + [self.stoi["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        color, shape, position = self.records[idx]
        image = self._render(color, shape, position)
        tokens = self._tokenize(color, shape, position)
        return image, tokens

    def decode(self, ids):
        words = []
        for i in ids.tolist():
            w = self.itos[i]
            if w == "<eos>":
                break
            if w in ("<pad>", "<bos>"):
                continue
            words.append(w)
        return " ".join(words)


# ---------------------------------------------------------------------------
# 真实数据集接入示例（伪代码，供你换成 Flickr8k/COCO 时参考，不在 sandbox 里运行）
# ---------------------------------------------------------------------------
"""
from torchvision import transforms
from PIL import Image
import json

class Flickr8kDataset(Dataset):
    def __init__(self, ann_file, img_dir, tokenizer, img_size=224, max_len=32):
        # ann_file: captions.json，每条包含 {"image": "xxx.jpg", "caption": "a dog runs..."}
        with open(ann_file) as f:
            self.records = json.load(f)
        self.img_dir = img_dir
        self.tokenizer = tokenizer  # 换成 BPE/wordpiece tokenizer（比如 tokenizers 库）
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(f"{self.img_dir}/{rec['image']}").convert("RGB")
        image = self.transform(image)
        tokens = self.tokenizer.encode(rec["caption"], max_len=self.max_len)
        return image, tokens
"""
