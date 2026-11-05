# -*- coding: utf-8 -*-
"""
长对话上下文控制（D-03）

最长对话达 50 轮，直接整段拼接可能超出模型上下文窗口。
本模块提供对话截断/滑动窗口工具，供数据工程（prepare_dataset.py）与推理链路复用：

- truncate_dialog：按「保留最近 N 轮 + 可选保留对话开头摘要」截断；
- render_turns：把 turns 渲染为文本，与推理端 conversation_text 保持一致。

策略（滑动窗口 + 头尾保留）：
- 始终保留最新 max_recent 轮（预测点所在位置附近的最近上下文最关键）；
- 若轮数超限，保留最前面 keep_head 轮（用户画像/背景信息常在开头）；
- 中间被丢弃的部分以占位提示标记，保证语义可读。

用法：
    from src.data.context import truncate_turns
    kept = truncate_turns(turns, max_recent=20, keep_head=2)
"""
from __future__ import annotations


def render_turns(turns: list[dict]) -> list[str]:
    """把 turns 列表渲染为 ['role: content', ...] 文本行。"""
    lines = []
    for t in turns:
        role = t.get("role", "")
        content = t.get("content", "")
        lines.append(f"{role}: {content}")
    return lines


def truncate_turns(turns: list[dict], max_recent: int = 20, keep_head: int = 2) -> list[dict]:
    """对 turns 列表做滑动窗口截断。

    参数：
        turns: 按时间顺序的轮次列表（每项含 role/content/turn_id 等）。
        max_recent: 最多保留的最近轮次数。
        keep_head: 最多保留的开头轮次数（画像/背景常在前几轮）。

    返回：截断后的 turns 列表（保持原始顺序）。
    逻辑：
        - 若总轮数 <= max_recent + keep_head，原样返回；
        - 否则保留「开头 keep_head 轮 + 结尾 max_recent 轮」，中间丢弃。
    """
    if len(turns) <= max_recent + keep_head:
        return list(turns)
    head = turns[:keep_head]
    tail = turns[-max_recent:]
    return head + tail


def render_with_truncation(turns: list[dict], max_recent: int = 20, keep_head: int = 2,
                           omission_marker: str = "\n…（中间对话已省略）…\n") -> str:
    """渲染 turns 为文本，超限时截断并插入省略标记。"""
    kept = truncate_turns(turns, max_recent=max_recent, keep_head=keep_head)
    lines = render_turns(kept)
    if len(kept) < len(turns):
        # 在 head 与 tail 交界处插入省略标记
        if keep_head > 0:
            head_lines = lines[:keep_head]
            tail_lines = lines[keep_head:]
            return "\n".join(head_lines) + omission_marker + "\n".join(tail_lines)
        return "\n".join(lines)
    return "\n".join(lines)


if __name__ == "__main__":
    # 简单自测
    turns = [{"turn_id": i, "role": "user" if i % 2 else "assistant", "content": f"消息{i}"}
             for i in range(1, 51)]
    kept = truncate_turns(turns, max_recent=20, keep_head=2)
    print(f"原始 {len(turns)} 轮 -> 截断后 {len(kept)} 轮（head {2} + tail {20}）")
    assert len(kept) == 22
    assert kept[0]["turn_id"] == 1 and kept[1]["turn_id"] == 2  # head
    assert kept[-1]["turn_id"] == 50  # tail 结尾
    text = render_with_truncation(turns)
    assert "中间对话已省略" in text
    print("自测通过")
