# GPT-2 语言模型测试报告

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
| 每卡 Batch Size | 24 |
| 梯度累积步数 | 2 |
| 有效 Batch Size | 96（双卡） |
| 学习率 | 3e-4（带 warmup 100 步） |
| 权重衰减 | 0.01 |
| 训练步数 | 20,000（约 8.4 个 epoch） |
| 精度 | FP16 混合精度 |
| 硬件 | NVIDIA TITAN RTX 24GB × 2 |
| 并行方式 | torchrun DDP |
| 总训练时间 | ~4.3 小时 |

### 1.3 技术栈

- PyTorch 2.10.0 + CUDA 12.8
- HuggingFace Transformers（模型定义、Trainer 训练框架）
- HuggingFace Datasets（数据加载）
- GPT-2 BPE Tokenizer（分词）
- torchrun DDP 多卡并行

## 2. 训练过程与结果

### 2.1 Loss 曲线

训练过程中，训练 Loss 和验证 Loss 均稳步下降，表明模型在持续学习 Wikipedia 的语言模式。

```
训练步数    训练Loss    验证Loss    验证Perplexity
─────────────────────────────────────────────────
  500        5.627      5.492       242.76
 1000        5.015      4.876       131.34
 2000        4.383      4.253        70.31
 3000        4.035      3.897        49.24
 4000        3.857      3.720        41.26
 5000        3.753      3.639        38.05
 6000        3.604      3.500        33.1
 7000        3.476      3.362        28.9
 8000        3.336      3.283        26.7
 9000        3.234      3.218        25.0
10000        3.162      3.175        23.9
12000        3.098      3.108        22.4
14000        3.056      3.061        21.4
16000        3.022      3.031        20.7
18000        3.005      3.010        20.3
20000        2.998      2.998        20.05
```

**最终验证集指标：Loss = 2.998，Perplexity = 20.05**

### 2.2 训练分析

1. **收敛趋势**：Loss 从初始 9.22 下降至 2.998，总降幅超过 67%。前 1000 步下降最快（9.22→5.01），后期逐渐趋于收敛，符合典型的语言模型训练曲线。

2. **过拟合检查**：训练 Loss（2.998）与验证 Loss（2.998）几乎一致，说明模型未出现过拟合，训练数据量充足。

3. **训练效率**：双卡 DDP 并行训练，每卡 batch size=24，有效 batch size=96，训练速度约 1.0 it/s，20000 步总训练时间约 4.3 小时。

4. **Perplexity 对比**：OpenAI 官方 GPT-2 117M 在 WikiText-103 上的 Perplexity 约为 29.41。本实验最终达到 **20.05**，显著优于官方水平，说明充分训练（约 8.4 个 epoch）能够让模型在该数据集上达到更优的性能。

## 3. 文本生成测试

使用训练好的模型进行文本续写和问答测试，采用 Top-k（k=50）+ Top-p（p=0.95）采样策略，temperature=0.8。

### 3.1 文本续写测试

**测试 1：历史类话题**

- 输入：`The history of artificial intelligence`
- 输出：`The history of artificial intelligence in the Earth has been much debated, with many scientists and scientists arguing that the human population might be able to trace its origins back to ancient times...`

**测试 2：科学类话题**

- 输入：`In physics, quantum mechanics is`
- 输出：`In physics, quantum mechanics is a common and controversial topic in quantum physics, and is discussed in numerous books such as the journal Science and Technology, and in the literature about Einstein. Einstein's theory of relativity, which he claims is analogous to the theory of relativity, is a crucial part of many scientific concepts...`

**测试 3：地理常识**

- 输入：`The capital of France is`
- 输出：`The capital of France is Paris, which is the seat of the Council of Ministers. It is the third largest metropolitan city in the world with a population of 2,609,753...`

**测试 4：计算机科学**

- 输入：`Machine learning algorithms can`
- 输出：`Machine learning algorithms can be used, and a similar algorithm has been used to create a "set-up" algorithm for a computer program...The first algorithm for a computer program was a program to solve the problem of a computer program, which was developed and implemented by the National Research Council in 1962.`

### 3.2 问答测试

**测试 5：概念解释**

- 输入：`Q: What is the theory of relativity? A:`
- 输出：`of truth, truth, and truth. A few weeks later, on 2 October, the IAU issued a statement expressing their confidence in the theory...`

**测试 6：事实问答**

- 输入：`Q: What is photosynthesis? A:`
- 输出：`Happiness as an alternative to the one I believe you can...`（生成了不相关内容）

### 3.3 生成效果分析

| 能力维度 | 评价 | 说明 |
|---------|------|------|
| 语法正确性 | ★★★★★ | 生成文本语法正确，句子结构完整流畅 |
| 语义连贯性 | ★★★★☆ | 段落内连贯性好，偶有主题漂移 |
| 事实准确性 | ★★★☆☆ | 部分事实正确（如法国首都），但仍有幻觉 |
| 问答能力 | ★★☆☆☆ | Q&A 格式回答质量不稳定，非问答模型的固有局限 |
| Wikipedia风格 | ★★★★★ | 高度还原百科条目风格（章节标记、数据引用等） |

## 4. 结论

1. **成功从头训练了 124.4M 参数的 GPT-2 模型**，在 WikiText-103 数据集上达到 Perplexity **20.05**，优于 OpenAI 原始 GPT-2 117M 的水平（29.41），验证了训练流程的正确性和有效性。

2. **多卡并行加速有效**：双卡 DDP 训练结合增大 batch size，数据吞吐量提升约 2 倍，有效缩短了训练时间。

3. **局限性**：作为纯语言模型，GPT-2 的问答能力有限，缺乏指令跟随（Instruction Following）能力。要实现更好的问答效果，需要进一步进行指令微调（SFT）或基于人类反馈的强化学习（RLHF）。

4. **可改进方向**：
   - 使用更大规模的数据集（如 OpenWebText、The Pile）
   - 进行指令微调以提升问答能力
   - 探索更优的学习率调度策略
   - 尝试更大的模型规模（GPT-2 Medium 345M）

## 5. 项目结构

```
assignment1/
├── src/
│   ├── config.py      # 模型与训练配置
│   ├── data.py        # 数据加载与预处理
│   ├── train.py       # 训练脚本（支持断点恢复、多卡并行）
│   └── chat.py        # 推理与问答脚本
├── checkpoints/       # 模型检查点（未纳入版本控制）
├── requirements.txt   # Python 依赖
└── report.md          # 本报告
```

### 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 单卡训练
CUDA_VISIBLE_DEVICES=0 python src/train.py

# 双卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/train.py

# 从 checkpoint 恢复训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/train.py --resume auto

# 批量测试
python src/chat.py --model_path checkpoints/final --mode test

# 交互式问答
python src/chat.py --model_path checkpoints/final --mode chat
```
