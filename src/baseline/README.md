# 基线系统（Baseline）

> 目标：用「基座模型 + 提示词 zero-shot」最快跑通 `start.sh`，产出首个**合法** `submission.jsonl`。

## 现状

基线已由 `participant/run_inference.py` 完整实现，无需额外代码：

- **提示词**：内置系统提示（见 `run_inference.py` 的 `SYSTEM_PROMPT`），注入 16 类情绪与三维画像枚举；
- **零样本**：不做任何微调，直接加载基座 CausalLM（transformers）或调用 vLLM 服务（openai_compatible）；
- **合法输出**：输出经 `extract_object → normalize_prediction` 清洗，字段越界自动降级为空，保证符合 `submission_schema.json`。

## 冒烟测试

```bash
# 用样例数据跑通，验证产物合法
bash participant/start.sh test_inference_data.jsonl result
python src/serve/json_output.py --submission result/submission.jsonl
```

期望：`json_output.py` 输出 `pass_rate: 1.0`（或记录越界降级，但不抛错）。

## 与训练版的关系

| 版本 | 推理路径 | 说明 |
|------|---------|------|
| 基线（本目录） | `run_inference.py` 内置 zero-shot 提示词 | 第 1 个合法提交，建立性能/分数基线 |
| 微调版 | `src/train/train.py` 产出权重 → 放入 `participant/model` | 在基线基础上迭代效果 |

基线的四项指标（验证集）作为后续微调的对照锚点，由 `src/eval/score.py` 计算。
