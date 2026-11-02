# -*- coding: utf-8 -*-
"""
阶段一/二 · 多任务微调（M9 训练构件）

将「逐预测点样本」（由 src/data/prepare_dataset.py 产出）微调为
「回复生成 + 情绪分类 + 用户画像多标签」三任务联合模型。

设计要点（对应系统设计文档 ADR-004 / 5.5 节）：
- 采用 instruction 式文本生成范式：一个样本渲染为「对话历史 + 指令」输入，
  期望输出为结构化 JSON（含 response_text / emotion_label / user_profile / memory_refs）。
  这与推理端 run_inference.py 的系统提示词 + 输出契约保持一致，保证训练/推理分布对齐。
- 情绪长尾处理：可选按训练集类别频率倒数计算加权（--emotion_weighting），
  缓解 anxiety 34.0% vs joy 0.9% 的极端不均衡。
- LoRA 高效微调（默认），显存受限可叠加 4bit 量化（--load_in_4bit）。

用法：
    python src/train/train.py \
        --train data/processed/train_samples.jsonl \
        --val data/processed/val_samples.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output_dir output/checkpoints/run1

注：本脚本为骨架，依赖 transformers + peft + datasets；模型名/路径需替换为实际可用基座。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

INSTRUCTION = (
    "你正在参加数字情感陪伴比赛。根据截止到最新用户消息的对话历史，完成下一轮 Assistant 回复预测。\n"
    "只输出一个 JSON object，不要输出 Markdown 或解释。object 必须且只能包含：\n"
    "- response_text: 非空中文回复；\n"
    "- emotion_label: 最新用户消息的情绪，从 16 类枚举中选择；\n"
    "- user_profile: object，含 personality_traits、interests、style 三个字符串数组；\n"
    "- memory_refs: 数组（测试集无公开记忆库时输出 []）。\n"
)


def load_samples(path: str):
    """逐行读取逐预测点样本，返回 list[dict]。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def render_prompt(sample: dict) -> str:
    """把逐预测点样本渲染为 instruction 格式的输入文本（与推理端一致）。"""
    return INSTRUCTION + "对话历史:\n" + sample["history"] + "\n"


def render_target(sample: dict) -> str:
    """把样本真值渲染为期望的 JSON 输出文本。"""
    target = {
        "response_text": sample["response_text"],
        "emotion_label": sample["emotion_label"],
        "user_profile": sample["user_profile"],
        "memory_refs": sample["memory_refs"],
    }
    return json.dumps(target, ensure_ascii=False)


def emotion_weights(samples, smoothing: float = 1.0):
    """按情绪类别频率倒数（平滑）计算类别权重，用于加权损失。"""
    counter = Counter(s["emotion_label"] for s in samples)
    total = sum(counter.values())
    n_class = len(counter)
    weights = {
        c: (total / (n_class * (cnt + smoothing)))
        for c, cnt in counter.items()
    }
    return weights


def build_datasets(train_samples, val_samples, tokenizer, max_len: int = 2048):
    """将样本转换为 datasets.Dataset（prompt + target 文本），供 HF Trainer 使用。"""
    from datasets import Dataset

    def to_hf(rows):
        return Dataset.from_list([
            {
                "prompt": render_prompt(s),
                "target": render_target(s),
            }
            for s in rows
        ])

    def tokenize(ex):
        full = [p + t for p, t in zip(ex["prompt"], ex["target"])]
        enc = tokenizer(full, truncation=True, max_length=max_len, padding=False)
        # 简单因果语言建模：labels 直接复用 input_ids
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_ds = to_hf(train_samples).map(tokenize, batched=True, remove_columns=["prompt", "target"])
    val_ds = to_hf(val_samples).map(tokenize, batched=True, remove_columns=["prompt", "target"])
    return train_ds, val_ds


def main():
    ap = argparse.ArgumentParser(description="多任务微调（M9 训练构件骨架）")
    ap.add_argument("--train", required=True, help="训练集逐预测点 jsonl")
    ap.add_argument("--val", required=True, help="验证集逐预测点 jsonl")
    ap.add_argument("--model", required=True, help="基座模型名或本地路径")
    ap.add_argument("--output_dir", required=True, help="检查点输出目录")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--use_lora", action="store_true", default=True, help="默认启用 LoRA")
    ap.add_argument("--load_in_4bit", action="store_true", help="4bit 量化")
    ap.add_argument("--emotion_weighting", action="store_true", help="打印情绪类别权重供参考")
    ap.add_argument("--save_steps", type=int, default=200)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

    train_samples = load_samples(args.train)
    val_samples = load_samples(args.val)
    print(f"训练样本 {len(train_samples)}，验证样本 {len(val_samples)}")

    if args.emotion_weighting:
        w = emotion_weights(train_samples)
        print("情绪类别权重（频率倒数平滑）：")
        for c in sorted(w):
            print(f"  {c}: {w[c]:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, **kwargs
    )

    if args.use_lora and not args.load_in_4bit:
        try:
            from peft import LoraConfig, get_peft_model
            lora = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora)
            model.print_trainable_parameters()
        except ImportError:
            print("[warn] 未安装 peft，跳过 LoRA，改全参微调")

    train_ds, val_ds = build_datasets(train_samples, val_samples, tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，权重已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
