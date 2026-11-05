# -*- coding: utf-8 -*-
"""
阶段一 · 数据工程
把 train/val 的公开 jsonl（prediction_points 嵌套结构）展开成"逐预测点"的训练样本，
便于后续多任务微调与评测对齐。

用法：
    python src/data/prepare_dataset.py \
        --src 训练-验证-数据集/train/train_public.jsonl \
        --out data/processed/train_samples.jsonl \
        [--max_turns 40]   # 可选：长对话截断（D-03），默认 0 表示不截断

输出每行一个 JSON：
    {
      "conversation_id", "target_user_turn_id", "assistant_response_turn_id",
      "history", "response_text", "emotion_label",
      "user_profile": {"personality_traits","interests","style"},
      "memory_refs": [...]
    }

说明：history 只保留到预测点 target_user_turn_id 之前的轮次（预测点之后的对话
不应出现在输入中，避免信息泄漏）。超长对话可用 --max_turns 触发滑动窗口截断。
"""
import argparse
import json
import os

from context import render_with_truncation


def build_history(turns, target_user_turn_id, max_turns=0):
    """渲染截至预测点前的对话历史；超长时截断（D-03）。"""
    # 只保留预测点之前的轮次（turn_id < target_user_turn_id）
    prior = [t for t in turns if t.get("turn_id", 0) < target_user_turn_id]
    if max_turns and max_turns > 0:
        return render_with_truncation(prior, max_recent=max_turns, keep_head=2)
    return "\n".join(f"{t['role']}: {t['content']}" for t in prior)


def flatten(src_path: str, out_path: str, max_turns: int = 0) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    truncated = 0
    with open(src_path, encoding="utf-8") as f, \
            open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d["conversation_id"]
            turns = d["turns"]
            for pp in d.get("prediction_points", []):
                tgt = pp["target"]
                tgt_turn = pp["target_user_turn_id"]
                history = build_history(turns, tgt_turn, max_turns)
                if max_turns and len([t for t in turns if t.get("turn_id", 0) < tgt_turn]) > max_turns + 2:
                    truncated += 1
                sample = {
                    "conversation_id": cid,
                    "target_user_turn_id": tgt_turn,
                    "assistant_response_turn_id": pp["assistant_response_turn_id"],
                    "history": history,
                    "response_text": tgt["response_text"],
                    "emotion_label": tgt["emotion_label"],
                    "user_profile": tgt["user_profile"],
                    "memory_refs": tgt["memory_refs"],
                }
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
    return n, truncated


def main():
    ap = argparse.ArgumentParser(description="展开竞赛数据集为逐预测点样本")
    ap.add_argument("--src", required=True, help="原始 train/val jsonl 路径")
    ap.add_argument("--out", required=True, help="展开后样本输出路径")
    ap.add_argument("--max_turns", type=int, default=0,
                    help="长对话截断阈值（0 表示不截断），超限保留头 2 轮 + 最近 N 轮")
    args = ap.parse_args()
    n, truncated = flatten(args.src, args.out, args.max_turns)
    print(f"已写出 {n} 个样本 -> {args.out}")
    if args.max_turns > 0:
        print(f"其中触发截断的预测点：{truncated} 个（阈值 {args.max_turns}）")


if __name__ == "__main__":
    main()
