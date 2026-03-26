"""GPT-2 训练配置"""

from dataclasses import dataclass


@dataclass
class GPT2Config:
    """GPT-2 117M 模型配置"""
    # 模型结构
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12

    # 训练超参数
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # 等效 batch_size = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 5000
    max_length: int = 512

    # 数据
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"

    # 保存与日志
    output_dir: str = "checkpoints"
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 500

    # 设备
    fp16: bool = True
