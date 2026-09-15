"""
把 eval 存下來的 agent trace 印成人看得懂的逐步紀錄,不用重新呼叫 LLM。

用途:診斷某一題為什麼答錯——看 LLM 每一步選了哪個工具、用什麼查詢字串、
拿到的信心分數、有沒有觸發 drift 介入、預算用到哪。README 裡好幾個根因診斷
(query drift、邊界名次抖動)都是靠這種逐步回放找到的,這支腳本把那個流程固定下來。

用法:
  python scripts/replay_trace.py eval/results_tool_agent_drift_check.json            # 列出所有題目
  python scripts/replay_trace.py eval/results_tool_agent_drift_check.json --miss     # 只看答錯的
  python scripts/replay_trace.py eval/results_tool_agent_drift_check.json --grep 紅麴
"""
import argparse
import json
from pathlib import Path


def iter_records(data: dict):
    for group in ("main", "hard", "easy"):
        for rec in data.get(group, {}).get("per_question", []):
            yield group, rec


def render(group: str, rec: dict) -> str:
    lines = [f"[{group}] hit={rec.get('hit')}  grounded={rec.get('grounded')}  "
             f"calls={rec.get('tool_calls_used')}  {rec.get('elapsed_s', '?')}s"]
    lines.append(f"  Q: {rec['question']}")
    if "budget" in rec:
        b = rec["budget"]
        lines.append(f"  budget: {b}")
    for i, step in enumerate(rec.get("trace", []), 1):
        args = json.dumps(step.get("arguments", {}), ensure_ascii=False)
        extra = ""
        if step.get("confident") is not None:
            extra += f"  confident={step['confident']} score={step.get('top_score')}"
        if step.get("query_drift_detected"):
            extra += "  ⚠drift→改用字面問題"
        lines.append(f"  {i}. {step['name']}({args}){extra}")
        if step.get("chunk_ids"):
            lines.append(f"     → chunks {step['chunk_ids']}")
    if not rec.get("trace") and rec.get("tools"):
        lines.append(f"  tools: {' > '.join(rec['tools'])}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--miss", action="store_true", help="只顯示 hit=False 的題目")
    ap.add_argument("--grep", default=None, help="只顯示問題含此字串的題目")
    args = ap.parse_args()

    data = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    shown = 0
    for group, rec in iter_records(data):
        if args.miss and rec.get("hit"):
            continue
        if args.grep and args.grep not in rec["question"]:
            continue
        print(render(group, rec))
        print()
        shown += 1
    print(f"({shown} 題)")


if __name__ == "__main__":
    main()
