"""
把「手寫 24 題」+「LLM 生成後人工審查通過的題目」合併成 eval_questions_v2.json。

人工審查的決定寫死在這個檔案裡(不是散在筆記裡),任何人都能重現同一份評估集,
也看得到哪些題目為什麼被剔掉。審查標準:
  - gold 必須唯一:問法籠統到很多 chunk 都能回答的(「原產地怎麼標示才正確」)剔掉
  - 表格切片剔掉:農藥殘留容許量表被 chunker 依長度切成多片,同一支農藥的不同作物
    散在相鄰 chunk,「布瑞莫對哪些水果適用」的答案跨 chunk,gold 不唯一
  - 檢驗方法 SOP 剔掉:「1 N 氫氧化鈉怎麼調」「玻璃滴管多長」在幾十份檢驗方法裡
    都有一模一樣的句子
  - OCR 亂碼、公文抬頭(「保存年限:」)、LLM 自己發明的詞(「螳螂藥」)剔掉
  - answer_span 沒有真的回答 question 的剔掉

用法:python eval/build_eval_v2.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

HERE = Path(__file__).parent

# 第一批(generated_questions.json,seed=42):80 題生成 → 人工審查留 48 題
BATCH1_KEEP = [
    1, 2, 3, 6, 7, 11, 12, 15, 16, 19, 21, 22, 25, 26, 28, 29, 30, 31, 33, 38,
    41, 42, 43, 45, 46, 48, 49, 51, 52, 53, 55, 56, 57, 60, 61, 63, 65, 66, 67, 68,
    69, 70, 72, 74, 75, 77, 78, 79,
]
BATCH1_REJECT_REASONS = {
    "gold 不唯一(問法太籠統)": [0, 4, 5, 10, 18, 34, 47, 50, 76],
    "農藥殘留表格切片,答案跨 chunk": [8, 9, 14, 39, 54, 64],
    "檢驗方法 SOP,多份文件有相同句子": [17, 37, 40],
    "chunk 是 tariff/統計表格列,問題與內容不符": [13, 27, 36, 44, 58],
    "OCR 亂碼或公文抬頭": [20, 23, 32, 35, 59, 62, 71, 73],
    "問題語意不通": [24],
}

# 第二批(generated_questions_b2.json,seed=7,配額偏向 qa)
BATCH2_KEEP = [
    0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 16, 17, 18, 19, 24, 26, 27, 29, 30, 32,
    34, 37, 38, 39, 40, 41, 42, 43, 44, 48, 49, 50, 53, 54, 58, 59,
]
BATCH2_REJECT_REASONS = {
    "gold 不唯一(問法太籠統)": [6, 13, 14, 21, 36, 45, 46, 47, 55, 56],
    "農藥/動物用藥殘留表格切片,答案跨 chunk": [4, 22, 35],
    "檢驗方法 SOP,多份文件有相同句子": [8, 25, 33, 51],
    "chunk 是 tariff/表格列或檢核表模板,問題與內容不符": [23, 31, 52],
    "OCR 亂碼或公文抬頭": [20, 28, 57],
    "gold chunk 與第一批重複(id=15914)": [15],
}


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def check_partition(n, keep, reasons, label):
    rejected = [i for ids in reasons.values() for i in ids]
    assert not set(keep) & set(rejected), f"{label}: 同一題既保留又剔除"
    missing = set(range(n)) - set(keep) - set(rejected)
    assert not missing, f"{label}: 未審查 {sorted(missing)}"


def main():
    manual = load("eval_questions.json")
    for q in manual:
        q["source"] = "manual"

    synthetic = []
    for fname, keep, reasons in (
        ("generated_questions.json", BATCH1_KEEP, BATCH1_REJECT_REASONS),
        ("generated_questions_b2.json", BATCH2_KEEP, BATCH2_REJECT_REASONS),
    ):
        if not (HERE / fname).exists():
            continue
        batch = load(fname)
        check_partition(len(batch), keep, reasons, fname)
        for i in keep:
            q = batch[i]
            synthetic.append({
                "question": q["question"], "gold_chunk_id": q["gold_chunk_id"],
                "source": "synthetic", "kind": q["kind"], "batch": fname,
            })

    golds = [q["gold_chunk_id"] for q in manual + synthetic]
    assert len(golds) == len(set(golds)), "gold chunk 重複"

    out = manual + synthetic
    (HERE / "eval_questions_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    kinds = {}
    for q in synthetic:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    print(f"eval_questions_v2.json: {len(out)} 題 = manual {len(manual)} + synthetic {len(synthetic)} {kinds}")


if __name__ == "__main__":
    main()
