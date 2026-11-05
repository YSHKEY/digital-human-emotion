# -*- coding: utf-8 -*-
"""
契约校验与 JSON 输出辅助（M4 契约校验构件 + M3 输出规范化的复用）

职责：
- 对提交/预测结果做 submission_schema.json 全量校验（M4）；
- 提供与推理端 run_inference.py 一致的枚举白名单（M3），供训练/评测/离线校验复用，
  避免多处硬编码枚举不一致。

用法：
    # 独立校验一份 submission.jsonl
    python src/serve/json_output.py --submission result/submission.jsonl

    # 作为库导入
    from src.serve.json_output import validate_submission_file, SUBMISSION_SCHEMA
"""
from __future__ import annotations

import argparse
import json
import re

# 与 submission_schema.json 完全一致的枚举白名单（单一事实来源）
EMOTIONS = {
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
}
PROFILE_LABELS = {
    "personality_traits": {"extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual", "agreeable", "assertive", "emotionally_stable", "sensitive"},
    "interests": {"study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design"},
    "style": {"brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user"},
}
MEMORY_RE = re.compile(r"^mem_[0-9]{6}$")
# 默认 id 前缀为 test_（公开样例与公开测试集）；隐藏集前缀不同时可通过
# ID_PREFIX 环境变量或 --id-prefix 参数覆盖（D-06）。
DEFAULT_ID_PREFIX = "test_"
ID_RE = re.compile(r"^test_[0-9]{6}_t[0-9]+$")

# 提交对象五字段契约（additionalProperties=false）
REQUIRED_FIELDS = {"id", "response_text", "emotion_label", "user_profile", "memory_refs"}


def make_id_re(prefix: str = DEFAULT_ID_PREFIX) -> "re.Pattern":
    """按前缀构造 id 校验正则（支持隐藏集前缀覆盖，D-06）。"""
    return re.compile(rf"^{re.escape(prefix)}[0-9]{{6}}_t[0-9]+$")


def validate_prediction(obj: dict, id_prefix: str = DEFAULT_ID_PREFIX) -> list[str]:
    """校验单条预测对象，返回错误列表（空列表表示通过）。"""
    errors = []
    if not isinstance(obj, dict):
        return ["不是 JSON object"]
    missing = REQUIRED_FIELDS - set(obj)
    if missing:
        errors.append(f"缺少字段: {sorted(missing)}")
    extra = set(obj) - REQUIRED_FIELDS
    if extra:
        errors.append(f"多余字段: {sorted(extra)}")

    id_re = make_id_re(id_prefix)
    if "id" in obj and not (isinstance(obj["id"], str) and id_re.fullmatch(obj["id"])):
        errors.append(f"id 格式非法: {obj.get('id')!r}")
    if "response_text" in obj and not isinstance(obj["response_text"], str):
        errors.append("response_text 不是字符串")
    if "emotion_label" in obj:
        e = obj["emotion_label"]
        if not (isinstance(e, str) and (e == "" or e in EMOTIONS)):
            errors.append(f"emotion_label 越界: {e!r}")
    if "user_profile" in obj:
        up = obj["user_profile"]
        if not isinstance(up, dict):
            errors.append("user_profile 不是 object")
        else:
            for k, allowed in PROFILE_LABELS.items():
                vals = up.get(k)
                if not (isinstance(vals, list)
                        and all(isinstance(v, str) for v in vals)
                        and len(vals) == len(set(vals))
                        and set(vals) <= allowed):
                    errors.append(f"user_profile.{k} 非法: {vals!r}")
    if "memory_refs" in obj:
        m = obj["memory_refs"]
        if not (isinstance(m, list)
                and all(isinstance(v, str) and MEMORY_RE.fullmatch(v) for v in m)
                and len(m) == len(set(m))):
            errors.append(f"memory_refs 非法: {m!r}")
    return errors


def validate_submission_file(path: str, id_prefix: str = DEFAULT_ID_PREFIX) -> dict:
    """校验整个 submission.jsonl，返回统计信息。"""
    total = 0
    bad = 0
    bad_rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                bad_rows.append({"line": i, "errors": [f"JSON 解析失败: {e.msg}"]})
                continue
            errs = validate_prediction(obj, id_prefix)
            if errs:
                bad += 1
                bad_rows.append({"line": i, "errors": errs})
    return {
        "total": total,
        "valid": total - bad,
        "invalid": bad,
        "pass_rate": round((total - bad) / total, 4) if total else 0.0,
        "bad_rows": bad_rows,
    }


def main():
    ap = argparse.ArgumentParser(description="submission_schema 契约校验（M4）")
    ap.add_argument("--submission", required=True, help="submission.jsonl 路径")
    ap.add_argument("--id-prefix", default=DEFAULT_ID_PREFIX,
                    help=f"id 前缀（默认 {DEFAULT_ID_PREFIX}，隐藏集可用 env ID_PREFIX 或本参数覆盖）")
    args = ap.parse_args()
    import os
    prefix = os.environ.get("ID_PREFIX", args.id_prefix)
    report = validate_submission_file(args.submission, prefix)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
