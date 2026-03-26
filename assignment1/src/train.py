"""GPT-2 从头训练脚本

支持从 checkpoint 恢复训练，支持多卡并行（通过 torchrun 或 accelerate launch）。
"""

import argparse
import os
import math
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Config as HFConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from config import GPT2Config
from data import load_and_tokenize


def build_model(config):
    """根据配置从头构建 GPT-2 模型"""
    hf_config = HFConfig(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    model = GPT2LMHeadModel(hf_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count / 1e6:.1f}M")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 恢复训练，传入路径或 'auto' 自动查找最新")
    args = parser.parse_args()

    config = GPT2Config()

    # 加载数据
    print("正在加载数据集...")
    dataset, tokenizer = load_and_tokenize(config)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    print(f"每卡 batch_size: {config.batch_size}, 梯度累积: {config.gradient_accumulation_steps}")
    print(f"最大训练步数: {config.max_steps}")

    # 构建模型
    model = build_model(config)

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        fp16=config.fp16 and torch.cuda.is_available(),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
    )

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 确定是否从 checkpoint 恢复
    resume_from = None
    if args.resume == "auto":
        # 自动查找最新 checkpoint
        if os.path.isdir(config.output_dir):
            ckpts = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-")]
            if ckpts:
                latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
                resume_from = os.path.join(config.output_dir, latest)
                print(f"自动恢复自: {resume_from}")
    elif args.resume:
        resume_from = args.resume
        print(f"恢复自: {resume_from}")

    # 开始训练
    print("开始训练...")
    trainer.train(resume_from_checkpoint=resume_from)

    # 保存最终模型
    final_dir = os.path.join(config.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"模型已保存至 {final_dir}")

    # 评估
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"验证集 Loss: {eval_results['eval_loss']:.4f}")
    print(f"验证集 Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
