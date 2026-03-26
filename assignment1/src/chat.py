"""基于训练好的 GPT-2 模型进行文本生成与问答"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_model(model_path):
    """加载训练好的模型和分词器"""
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8,
             top_k=50, top_p=0.95, num_return_sequences=1):
    """根据提示生成文本"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(text)
    return results


def interactive_chat(model, tokenizer):
    """交互式问答"""
    print("=" * 50)
    print("GPT-2 Wiki 问答系统")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        prompt = input("\n用户: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not prompt:
            continue

        # 构造问答格式的 prompt
        qa_prompt = f"Q: {prompt}\nA:"
        responses = generate(model, tokenizer, qa_prompt, max_new_tokens=150)

        for resp in responses:
            # 提取回答部分
            if "A:" in resp:
                answer = resp.split("A:", 1)[1].strip()
                # 截断到第一个完整句子结束
                for end_char in ["\n\n", "\nQ:", "\n\n\n"]:
                    if end_char in answer:
                        answer = answer[:answer.index(end_char)]
                print(f"GPT-2: {answer}")
            else:
                print(f"GPT-2: {resp}")


def batch_test(model, tokenizer):
    """批量测试文本生成效果"""
    test_prompts = [
        "The history of artificial intelligence",
        "In physics, quantum mechanics is",
        "The capital of France is",
        "Machine learning algorithms can",
        "Q: What is the theory of relativity?\nA:",
        "Q: Who invented the telephone?\nA:",
        "Q: What is photosynthesis?\nA:",
    ]

    print("=" * 60)
    print("批量生成测试")
    print("=" * 60)

    results = []
    for prompt in test_prompts:
        print(f"\n【输入】{prompt}")
        responses = generate(model, tokenizer, prompt, max_new_tokens=100)
        for resp in responses:
            print(f"【输出】{resp}")
        results.append({"prompt": prompt, "response": responses[0]})
        print("-" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="GPT-2 文本生成与问答")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/final",
        help="模型路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "test"],
        default="test",
        help="运行模式: chat=交互问答, test=批量测试",
    )
    args = parser.parse_args()

    print(f"正在加载模型: {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    print("模型加载完成！")

    if args.mode == "chat":
        interactive_chat(model, tokenizer)
    else:
        batch_test(model, tokenizer)


if __name__ == "__main__":
    main()
