"""数据加载与预处理"""

from datasets import load_dataset
from transformers import GPT2TokenizerFast


def get_tokenizer():
    """加载 GPT-2 分词器"""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_and_tokenize(config):
    """加载 wikitext 数据集并进行分词处理

    将文本拼接后按 max_length 切分为等长序列，用于语言模型训练。
    """
    tokenizer = get_tokenizer()

    dataset = load_dataset(config.dataset_name, config.dataset_config)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=True,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=8,
        remove_columns=["text"],
        desc="分词中",
    )

    # 将所有 token 拼接后按固定长度切分
    block_size = config.max_length

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [v[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=8,
        desc="分组中",
    )

    return lm_dataset, tokenizer
