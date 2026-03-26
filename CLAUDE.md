# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course project repository (AIT531028) for "基于大模型的电子系统设计自动化". Each assignment lives in its own `assignmentN/` directory.

## Assignment 1: GPT-2 Training (`assignment1/`)

Train a GPT-2-scale (124.4M parameter) transformer language model from scratch on WikiText-103.

### Commands

```bash
cd assignment1

# Install dependencies
pip install -r requirements.txt

# Train (single GPU)
CUDA_VISIBLE_DEVICES=0 python src/train.py

# Batch test
python src/chat.py --model_path checkpoints/final --mode test

# Interactive chat
python src/chat.py --model_path checkpoints/final --mode chat
```

### Architecture

- `src/config.py` — dataclass with all model/training hyperparameters
- `src/data.py` — loads WikiText via HuggingFace Datasets, tokenizes with GPT-2 BPE, groups into fixed-length blocks
- `src/train.py` — builds GPT2LMHeadModel from scratch config, trains with HF Trainer
- `src/chat.py` — inference script with batch test and interactive Q&A modes

Key: model is initialized randomly (not from pretrained weights). Uses HF Trainer with FP16, gradient accumulation, and warmup schedule.

## Language

This is a Chinese university course project. Use Chinese for comments, documentation, and reports unless the user specifies otherwise.
