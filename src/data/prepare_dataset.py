# -*- coding: utf-8 -*-
"""
阶段一 · 数据工程
把 train/val 的公开 jsonl（prediction_points 嵌套结构）展开成"逐预测点"的训练样本，
便于后续多任务微调与评测对齐。

用法：
    python src/data/prepare_dataset.py \
        --src 训练-验证-数据集/train/train_public.jsonl \
        --out data/processed/train_samples.jsonl

输出每行一个 JSON：
    {
      "conversation_id", "target_user_turn_id", "assistant_response_turn_id",
      "history", "response_text", "emotion_label",
      "user_profile": {"personality_traits","interests","style"},
      "memory_refs": [...]
    }
"""
import argparse
import json
import os


def flatten(src_path: str, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(src_path, encoding="utf-8") as f, \
            open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d["conversation_id"]
            history = "\n".join(
                f"{t['role']}: {t['content']}" for t in d["turns"]
            )
            for pp in d.get("prediction_points", []):
                tgt = pp["target"]
                sample = {
                    "conversation_id": cid,
                    "target_user_turn_id": pp["target_user_turn_id"],
                    "assistant_response_turn_id": pp["assistant_response_turn_id"],
                    "history": history,
                    "response_text": tgt["response_text"],
                    "emotion_label": tgt["emotion_label"],
                    "user_profile": tgt["user_profile"],
                    "memory_refs": tgt["memory_refs"],
                }
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="展开竞赛数据集为逐预测点样本")
    ap.add_argument("--src", required=True, help="原始 train/val jsonl 路径")
    ap.add_argument("--out", required=True, help="展开后样本输出路径")
    args = ap.parse_args()
    n = flatten(args.src, args.out)
    print(f"已写出 {n} 个样本 -> {args.out}")


if __name__ == "__main__":
    main()
