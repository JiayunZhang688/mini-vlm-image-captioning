# Mini-VLM：从零实现的小型视觉-语言模型（图像描述生成）

一个不依赖 `transformers`/`CLIP` 封装库、手写 Multi-Head Attention 的迷你 VLM，
用于验证并展示你对 **Self-Attention / Cross-Attention / ViT / Transformer Decoder**
的实现能力，对应"端到端大模型算法实习生"JD 里的：

> 熟悉 Transformer、ViT、LLM 等主流模型架构；具备 VLM 方向的研究或工程经验，
> 理解视觉-语言对齐、多模态建模等核心技术

## 项目结构

```
mini-vlm/
├── model.py                 # ViT 图像编码器 + Transformer 文本解码器（含手写 MHA/cross-attention）
├── dataset.py                # 离线合成数据集（几何图形+颜色+位置） + 真实数据集接入示例
├── train.py                   # 训练脚本
├── visualize_attention.py     # 生成描述 + 可视化 cross-attention 热力图
├── checkpoints/                # 训练好的权重
└── samples/                    # 可视化输出（面试展示用）
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

这一步的目的**不是**做出一个多牛的模型，而是证明整条 pipeline（图像切 patch → ViT 编码 →
causal self-attention 生成文本 → cross-attention 做视觉-语言对齐）你能独立写出来、
调得通、看得懂内部在发生什么——这是面试时最常被追问细节的部分。

## 换成真实数据集（写进简历前建议做这一步）

合成数据集只是用来验证代码正确性。真正能写进简历的版本，建议换成：

1. **Flickr8k**（约 8000 张图 + 5 条人工描述/图，体量适合几天内跑完）
   - Kaggle 搜 "Flickr8k Dataset" 下载
   - 参考 `dataset.py` 底部注释的 `Flickr8kDataset` 示例改写
   - 把 tokenizer 换成简单的 BPE（可以直接用 `tokenizers` 库训练一个小词表，
     正好对应你简历里已经写的 "BPE 分词" 经验）
2. `model.py` 完全不用改（接口是 image tensor + token id 序列，和合成数据集一致）
3. 把 `img_size` 调到 224、`patch_size` 调到 16（标准 ViT 配置），`d_model` 调到 256~384，
   在单张 GPU（Colab 免费T4即可）上跑几个 epoch
4. 用 BLEU-4 / CIDEr 做定量评估（`pycocoevalcap` 库），比"生成的句子看起来对"更有说服力

## 进阶方向（如果还想覆盖 RL 相关要求）

JD 里提到"了解强化学习基本原理，有 RL 在决策或控制任务中应用经验者优先"。
image captioning 领域有个经典技术叫 **Self-Critical Sequence Training (SCST)**——
用 REINFORCE 直接优化不可微的评估指标（如 CIDEr），本质就是一次简化版的 RL 后训练：

- Reward = 生成句子的 CIDEr 分数（sparse reward，整句结束才算）
- Baseline = 贪心解码的句子的 CIDEr 分数（降低方差，这是 SCST 的核心 trick）
- Loss = `-(reward - baseline) * log_prob(sampled_sentence)`

在 `train.py` 的 teacher-forcing 训练收敛后，加一个 SCST fine-tune 阶段（几十行代码），
是同时覆盖"VLM + RL"两块要求、性价比很高的一步——如果你决定投入更多时间，这是推荐的下一步。


- 为什么 attention 要除以 `sqrt(d_head)`（防止 softmax 饱和、梯度消失）
- self-attention 和 cross-attention 的 Q/K/V 分别来自哪里
- causal mask 为什么用加性 `-inf` 而不是乘性 0/1
- ViT 的 patch embedding 本质是一个 stride=patch_size 的卷积
