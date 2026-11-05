# -*- coding: utf-8 -*-
"""
验证集四项指标评测（对齐版）

对提交结果 submission.jsonl 与验证集真值（含 prediction_points 的 jsonl）逐预测点对齐，
计算四项指标：情绪 macro-F1、画像命中率、记忆命中率、回复文本长度统计。

对齐方式（修复 D-04）：
- 真值：把每段对话的 prediction_points 全部展开，按 (conversation_id, target_user_turn_id) 建索引；
- 提交：每行含 id（形如 test_XXXXXX_tN）与 conversation_id（若含）；
  优先用 id 中的 turn 序号 + 对话归属对齐，回退到按行序对齐。

用法：
    python src/eval/score.py \
        --gt 训练-验证-数据集/val/val_public.jsonl \
        --submission result/submission.jsonl
"""
import argparse
import json
import re
from collections import defaultdict

# id 形如 test_000001_t3 或 val_conv_000001（对话级）——提取 turn 序号（若存在）
TURN_RE = re.compile(r"_t(\d+)$")


def load_gt(path: str):
    """读取验证集真值，展开所有预测点，返回 (keys, targets) 列表。

    每个预测点 key = (conversation_id, target_user_turn_id)。
    """
    keys = []
    targets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d["conversation_id"]
            for pp in d.get("prediction_points", []):
                keys.append((cid, pp["target_user_turn_id"]))
                targets.append(pp["target"])
    return keys, targets


def load_submission(path: str):
    """读取提交，返回 list[dict]，保留 id 与 conversation_id 字段用于对齐。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def submission_keys(sub):
    """尝试从提交行提取 (conversation_id, turn_id)，无法提取则返回 None。"""
    keys = []
    for s in sub:
        cid = s.get("conversation_id")
        turn = None
        sid = s.get("id", "")
        m = TURN_RE.search(sid)
        if m:
            turn = int(m.group(1))
        if cid is not None and turn is not None:
            keys.append((cid, turn))
        else:
            keys.append(None)
    return keys


def align(gt_keys, gt_targets, sub):
    """将提交按 key 对齐到真值顺序，返回对齐后的 pred 列表与统计。

    优先按 (conversation_id, turn_id) 精确对齐；无法对齐时按行序回退。
    """
    sub_keys = submission_keys(sub)
    # 若所有提交行都能提取 key，则按 key 建索引对齐
    if all(k is not None for k in sub_keys):
        idx = {k: i for i, k in enumerate(sub_keys)}
        pred = []
        matched = 0
        for k in gt_keys:
            if k in idx:
                pred.append(sub[idx[k]])
                matched += 1
            else:
                pred.append(None)
        return pred, matched
    # 回退：按行序对齐
    return sub, min(len(gt_keys), len(sub))


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
    ap = argparse.ArgumentParser(description="验证集四项指标评测（多预测点对齐版）")
    ap.add_argument("--gt", required=True, help="验证集真值 jsonl（含 prediction_points）")
    ap.add_argument("--submission", required=True, help="模型提交 submission.jsonl")
    args = ap.parse_args()

    gt_keys, gt_targets = load_gt(args.gt)
    sub = load_submission(args.submission)

    pred, matched = align(gt_keys, gt_targets, sub)

    # 过滤掉未对齐的预测点（pred 为 None），只对可对齐部分计算指标
    gold_emotion = []
    pred_emotion = []
    gold_profile = []
    pred_profile = []
    gold_mem = []
    pred_mem = []
    pred_text_lens = []
    skipped = 0
    for g, p in zip(gt_targets, pred):
        if p is None:
            skipped += 1
            continue
        gold_emotion.append(g.get("emotion_label", ""))
        pred_emotion.append(p.get("emotion_label", "") or "")
        gold_profile.append(g.get("user_profile", {}) or {})
        pred_profile.append(p.get("user_profile", {}) or {})
        gold_mem.append(g.get("memory_refs", []) or [])
        pred_mem.append(p.get("memory_refs", []) or [])
        pred_text_lens.append(len(p.get("response_text", "") or ""))

    n = len(gold_emotion)
    report = {
        "gt_points": len(gt_targets),
        "aligned": n,
        "skipped": skipped,
        "emotion_macro_f1": round(macro_f1(gold_emotion, pred_emotion), 4),
        "emotion_accuracy": round(
            sum(1 for g, p in zip(gold_emotion, pred_emotion) if g == p) / n, 4
        ) if n else 0.0,
        "profile_exact_hit": round(set_hit_rate(gold_profile, pred_profile), 4),
        "memory_exact_hit": round(set_hit_rate(gold_mem, pred_mem), 4),
        "response_avg_len": round(sum(pred_text_lens) / n, 1) if n else 0.0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
