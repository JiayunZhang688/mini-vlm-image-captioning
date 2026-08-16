# Mini-VLM：从零实现的小型视觉-语言模型（图像描述生成）


## 项目结构

```
mini-vlm/
├── model.py                 # ViT 图像编码器 + Transformer 文本解码器（含手写 MHA/cross-attention）
├── dataset.py                # 离线合成数据集（几何图形+颜色+位置） + 真实数据集接入示例
├── train.py                   # 训练脚本
├── visualize_attention.py     # 生成描述 + 可视化 cross-attention 热力图
├── checkpoints/                # 训练好的权重
└── samples/                    # 可视化输出
```

## 已验证：跑得通、能收敛

在完全离线生成的合成数据集（纯色背景 + 随机颜色/形状/位置的几何图形，配模板文字描述）上：

```bash
pip install torch numpy pillow matplotlib
python train.py --epochs 20 --n_samples 3000
python visualize_attention.py --n 3
```

- 参数量约 1M，CPU 上几分钟即可训练完
- 20 epoch 后 loss 从 1.64 收敛到 0.0015，生成描述与真实描述完全一致
- `visualize_attention.py` 会画出模型在生成每个词时，cross-attention 更关注图像的哪个 patch

