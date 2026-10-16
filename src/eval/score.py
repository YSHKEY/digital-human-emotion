# -*- coding: utf-8 -*-
"""
阶段一 · 验证集评测骨架
对提交结果 submission.jsonl 与验证集真值（同样含 prediction_points 的 jsonl）逐行对齐，
计算四项指标：情绪 macro-F1、画像命中率、记忆命中率、回复文本长度统计。

说明：本地验证时，请把验证集整理成"测试输入文件"（每行带 id 与历史），
评测脚本读取该文件与 submission.jsonl 按行顺序对齐。本骨架按行序对齐，可后续改为按 (conversation_id, turn_id) 对齐。

用法：
    python src/eval/score.py \
        --gt 训练-验证-数据集/val/val_public.jsonl \
        --submission result/submission.jsonl
"""
import argparse
import json
from collections import defaultdict


def load_gt(path: str):
    """读取验证集真值，返回按行序的 target 列表。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # 取第一个预测点作为该行的真值（多预测点场景需自行展开）
            tgt = d["prediction_points"][0]["target"]
            rows.append(tgt)
    return rows


def load_submission(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def macro_f1(gold_labels, pred_labels):
    classes = set(gold_labels) | set(pred_labels)
    f1_sum = 0.0
    for c in classes:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_sum += f1
    return f1_sum / len(classes) if classes else 0.0


def set_hit_rate(gold_sets, pred_sets):
    if not gold_sets:
        return 0.0
    hit = sum(1 for g, p in zip(gold_sets, pred_sets) if set(g) == set(p))
    return hit / len(gold_sets)


def main():
    ap = argparse.ArgumentParser(description="验证集四项指标评测骨架")
    ap.add_argument("--gt", required=True, help="验证集真值 jsonl（含 prediction_points）")
    ap.add_argument("--submission", required=True, help="模型提交 submission.jsonl")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    sub = load_submission(args.submission)
    n = min(len(gt), len(sub))
    if len(gt) != len(sub):
        print(f"[warn] 行数不一致：gt={len(gt)} submission={len(sub)}，按前 {n} 行对齐")

    gold_emotion = [g["emotion_label"] for g in gt[:n]]
    pred_emotion = [s.get("emotion_label", "") or "" for s in sub[:n]]

    gold_profile = [g["user_profile"] for g in gt[:n]]
    pred_profile = [s.get("user_profile", {}) or {} for s in sub[:n]]

    gold_mem = [g["memory_refs"] for g in gt[:n]]
    pred_mem = [s.get("memory_refs", []) or [] for s in sub[:n]]

    report = {
        "samples": n,
        "emotion_macro_f1": round(macro_f1(gold_emotion, pred_emotion), 4),
        "emotion_accuracy": round(
            sum(1 for g, p in zip(gold_emotion, pred_emotion) if g == p) / n, 4
        ) if n else 0.0,
        "profile_exact_hit": round(set_hit_rate(
            [p for p in gold_profile], [p for p in pred_profile]), 4),
        "memory_exact_hit": round(set_hit_rate(gold_mem, pred_mem), 4),
        "response_avg_len": round(
            sum(len(s.get("response_text", "") or "") for s in sub[:n]) / n, 1) if n else 0.0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
