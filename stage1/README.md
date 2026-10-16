# 阶段一（W1–W6）开发计划与代码骨架

> 阶段一目标：**先有能跑通评测的版本，再迭代**。第 6 周末前要在验证集上出第一个分数。

## 本阶段代码（骨架，待三人分工填充）

| 路径 | 负责人 | 内容 |
|------|--------|------|
| `src/data/prepare_dataset.py` | 成员A | 把 `train/val` 公开 jsonl 展开成逐预测点训练样本 |
| `src/eval/score.py` | 成员C | 验证集四项指标评测（情绪 macro-F1 / 画像命中 / 记忆命中 / 回复长度） |
| `src/baseline/` | 成员B | 基线：基座模型 + 提示词 zero-shot 跑通 `start.sh`，产出首个合法 `submission.jsonl` |
| `src/train/` | 成员A | 多任务微调脚本（LoRA/全参）、情绪长尾加权 |
| `src/serve/` | 成员B | 受约束解码 / JSON 输出控制、`submission_schema` 校验 |

## 阶段一待办（勾选推进）

- [ ] 数据工程：展开样本、统计情绪分布、确定训练/校验切分
- [ ] 基线系统：用 `participant/start.sh` 跑通 `test_inference_data.jsonl`，产出合法 submission
- [ ] 评测框架：本地 `score.py` 接入验证集，建立基线分数看板
- [ ] 接口联调：`submission_schema.json` 全字段合规，单条失败不中断整批
- [ ] 仓库与协作：GitHub/Gitee 远程仓库建立，三人分支与 PR 流程跑通

## 数据说明

- 训练集 `train_public.jsonl` / 验证集 `val_public.jsonl` 已 `.gitignore`（体积大、且各成员从竞赛平台获取一致），
  请在本机对应路径放置数据后运行脚本。
- 小样例 `test_inference_data.jsonl`（3 条）随仓库保留，用于冒烟测试。
