# GPT-2 语言模型从头训练 —— 测试报告

> 课程：AIT531028 基于大模型的电子系统设计自动化
>
> 作业一：LLM模型的训练

## 1. 实验概述

本实验从头（from scratch）训练了一个 GPT-2 规模的 Transformer 语言模型，使用 Wikipedia 语料作为训练数据，验证模型的语言建模能力和文本生成效果。

### 1.1 模型配置

| 参数 | 值 |
|------|-----|
| 模型架构 | GPT-2 (Decoder-only Transformer) |
| 参数量 | 124.4M |
| 隐藏层维度 (n_embd) | 768 |
| Transformer 层数 (n_layer) | 12 |
| 注意力头数 (n_head) | 12 |
| 最大序列长度 (n_positions) | 1024 |
| 词表大小 (vocab_size) | 50257 |

### 1.2 训练配置

| 参数 | 值 |
|------|-----|
| 训练数据 | Salesforce/wikitext (wikitext-103-raw-v1) |
| 训练样本数 | 229,395 条（按512 token分块后） |
| 验证样本数 | 481 条 |
| Batch Size | 8 × 4（梯度累积） = 有效 batch 32 |
| 学习率 | 5e-4（带 warmup 200 步） |
| 权重衰减 | 0.01 |
| 训练步数 | 5,000 |
| 精度 | FP16 混合精度 |
| 硬件 | NVIDIA TITAN RTX 24GB × 1 |

### 1.3 技术栈

- PyTorch 2.10.0 + CUDA 12.8
- HuggingFace Transformers（模型定义、Trainer 训练框架）
- HuggingFace Datasets（数据加载）
- GPT-2 BPE Tokenizer（分词）

## 2. 训练过程与结果

### 2.1 Loss 曲线

训练过程中，训练 Loss 和验证 Loss 均稳步下降，表明模型在持续学习 Wikipedia 的语言模式。

```
训练步数    训练Loss    验证Loss    验证Perplexity
─────────────────────────────────────────────────
   50        9.223        -            -
  500        5.627      5.492       242.76
 1000        5.015      4.876       131.34
 1500        4.654      4.511        91.05
 2000        4.383      4.253        70.31
 2500        4.162      4.033        56.41
 3000        4.035      3.897        49.24
 3500        3.922      3.798        44.39
 4000        3.857      3.720        41.26
 4500        3.797      3.667        39.13
 5000        3.753      3.639        38.05
```

**最终验证集指标：Loss = 3.639，Perplexity = 38.05**

### 2.2 训练分析

1. **收敛趋势**：Loss 从初始 9.22 下降至 3.75，降幅超过 59%。前 1000 步下降最快（9.22→5.01），后期逐渐趋于平缓，符合典型的语言模型训练曲线。

2. **过拟合检查**：训练 Loss（3.753）与验证 Loss（3.639）接近，验证 Loss 略低于训练 Loss，说明模型未出现过拟合。

3. **训练效率**：在单张 TITAN RTX 上，FP16 混合精度训练速度约 1.47 it/s，5000 步总训练时间约 57 分钟。

4. **Perplexity 对比**：OpenAI 官方 GPT-2 117M 在 WikiText-103 上的 Perplexity 约为 29.41（经过完整训练），本实验仅训练 5000 步即达到 38.05，说明模型学习效果良好，继续训练可进一步提升。

## 3. 文本生成测试

使用训练好的模型进行文本续写和问答测试，采用 Top-k（k=50）+ Top-p（p=0.95）采样策略，temperature=0.8。

### 3.1 文本续写测试

**测试 1：历史类话题**

- 输入：`The history of artificial intelligence`
- 输出：`The history of artificial intelligence was to prove the cause of the existence of humans. These theories of the scientific community have not been accepted by the scientific community...`

**测试 2：科学类话题**

- 输入：`In physics, quantum mechanics is`
- 输出：`In physics, quantum mechanics is the most common of these two-thirds of the unit's magnetism...`

**测试 3：地理常识**

- 输入：`The capital of France is`
- 输出：`The capital of France is now the country's largest capital, and it is the largest city in Europe. The city is the largest capital in the world, with a total of 13,000 units...`

### 3.2 问答测试

**测试 4：概念解释**

- 输入：`Q: What is the theory of relativity? A:`
- 输出：`in the field of relativity, a theory that was first studied by the United States in the late 1950s and early 1950s.`

**测试 5：事实问答**

- 输入：`Q: Who invented the telephone? A:`
- 输出：生成了不相关的内容（模型未能准确回答事实性问题）

### 3.3 生成效果分析

| 能力维度 | 评价 | 说明 |
|---------|------|------|
| 语法正确性 | ★★★★☆ | 生成文本语法基本正确，句子结构完整 |
| 语义连贯性 | ★★★☆☆ | 短句内连贯，但长文本会出现主题漂移 |
| 事实准确性 | ★★☆☆☆ | 会生成看似合理但不准确的信息（幻觉） |
| 问答能力 | ★★☆☆☆ | 能理解Q&A格式但回答质量不稳定 |
| Wikipedia风格 | ★★★★☆ | 生成文本带有明显的百科条目风格（章节标记等） |

## 4. 结论

1. **成功从头训练了 124.4M 参数的 GPT-2 模型**，在 WikiText-103 数据集上达到 Perplexity 38.05，接近 OpenAI 原始 GPT-2 117M 的水平（29.41）。

2. **模型具备基本的语言建模能力**，能够生成语法正确、风格符合 Wikipedia 的英文文本，证明 Transformer 架构在语言建模任务上的有效性。

3. **局限性**：作为纯语言模型，GPT-2 的问答能力有限，缺乏指令跟随（Instruction Following）能力。要实现更好的问答效果，需要进一步进行指令微调（SFT）或基于人类反馈的强化学习（RLHF）。

4. **可改进方向**：
   - 增加训练步数（当前仅训练 5000 步，不到 1 个 epoch）
   - 使用更大的数据集或混合多种语料
   - 进行指令微调以提升问答能力
   - 使用多卡并行加速训练

## 5. 项目结构

```
assignment1/
├── src/
│   ├── config.py      # 模型与训练配置
│   ├── data.py        # 数据加载与预处理
│   ├── train.py       # 训练脚本
│   └── chat.py        # 推理与问答脚本
├── checkpoints/       # 模型检查点（未纳入版本控制）
├── requirements.txt   # Python 依赖
└── report.md          # 本报告
```

### 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
CUDA_VISIBLE_DEVICES=0 python src/train.py

# 批量测试
python src/chat.py --model_path checkpoints/final --mode test

# 交互式问答
python src/chat.py --model_path checkpoints/final --mode chat
```
